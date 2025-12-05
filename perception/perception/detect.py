import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import WaypointArray, ObstacleArray, Obstacle as ObstacleMessage
import math
from bisect import bisect_left
from perception.frenet_converter import FrenetConverter
from tf_transformations import quaternion_matrix, quaternion_from_euler
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from builtin_interfaces.msg import Duration as DurationMsg
from scipy.spatial.transform import Rotation as R
from rclpy.time import Time


def normalize_s(x,track_length):
        x = x % (track_length)
        if x > track_length/2:
            x -= track_length
        return x


class Obstacle(object):
    current_id = 0
    def __init__(self,x,y,size,theta) -> None:
        self.center_x = x
        self.center_y = y
        self.size = size
        self.id = None
        self.theta = theta
    
    def squaredDist(self, obstacle):
        return (self.center_x-obstacle.center_x)**2+(self.center_y-obstacle.center_y)**2


class Detect(Node):
    def __init__(self):
        super().__init__('detect_node')

        # Parameters
        self.declare_parameter('min_points_per_cluster', 5)
        self.declare_parameter('enable_track_filter', True)
        self.declare_parameter('max_viewing_distance', 5.0)  #9
        self.declare_parameter('lambda_angle', 5.0 * math.pi / 180.0)  # 5 degrees
        self.declare_parameter('sigma', 0.01)  # standard deviation for adaptive clustering
        self.declare_parameter('min_obs_size', 5)
        self.declare_parameter('min_2_points_dist', 0.1)  # minimum distance between two points to be considered an obstacle
        self.declare_parameter('max_obs_size', 0.8)   # 10

        # Load parameters
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.min_points_per_cluster = self.get_parameter('min_points_per_cluster').get_parameter_value().integer_value
        self.enable_track_filter = self.get_parameter('enable_track_filter').get_parameter_value().bool_value
        self.max_viewing_distance = self.get_parameter('max_viewing_distance').get_parameter_value().double_value
        self.lambda_angle = self.get_parameter('lambda_angle').get_parameter_value().double_value
        self.sigma = self.get_parameter('sigma').get_parameter_value().double_value
        self.min_obs_size = self.get_parameter('min_obs_size').get_parameter_value().integer_value
        self.min_2_points_dist = self.get_parameter('min_2_points_dist').get_parameter_value().double_value
        self.max_obs_size = self.get_parameter('max_obs_size').get_parameter_value().double_value

        # marker_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=QoSDurabilityPolicy.VOLATILE,
        #     # history=HistoryPolicy.KEEP_LAST,
        #     depth=10
        # )

        # Publishers
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/markers', 10)   
        self.pub_boundaries = self.create_publisher(Marker, '/perception/boundaries', 10)
        self.pub_breakpoints_markers = self.create_publisher(MarkerArray, '/perception/breakpoints', 10)
        self.pub_obstacles_message = self.create_publisher(ObstacleArray, '/perception/raw_obstacles', 10)
        # self.pub_debug = self.create_publisher(MarkerArray, '/perception/track_debug', 10)
        # self.pub_object = self.create_publisher(MarkerArray, '/perception/object_markers', 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.laser_callback, sensor_qos)    # 10
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(WaypointArray, '/global_centerline', self.path_callback, 10)

        # States
        self.current_stamp = None
        self._debug_id = 0
        self.track_ready = False
        self.car_pose = None
        self.car_s = None
        self.waypoints = None
        self.s_array = None
        self.d_right_array = None
        self.d_left_array = None
        self.track_length = 0.0
        self.smallest_d = 0.0
        self.biggest_d = 0.0
        self.path_needs_update = False
        self.boundary_inflation = 0.25  # 0.1, 0.3
        self.H_map_bl = None
        self.t_map_bl = None  # rclpy.time.Time of the cached transform
        self.prev_ids = set()
        self.prev_ids_obs = set()

        self.ema_alpha = 0.6
        self.prev_center = {}   # id -> (x,y)

        self.scan = None
        self.tracked_obstacles = []

        # initialize frenet converter
        self.converter = None

    def laser_callback(self, scan):
        self.scan = scan
        self.detect()

    def odom_callback(self, odom):
        '''Get car pose and convert to Frenet coordinates.'''
        # print("odom callback")
        self.current_stamp = odom.header.stamp
        self.car_pose = odom.pose.pose
        x = self.car_pose.position.x
        y = self.car_pose.position.y
        q = self.car_pose.orientation
        # convert quaternion to yaw
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # get frenet coordinates
        if self.converter is None:
            self.get_logger().warn("FrenetConverter not initialized, cannot convert to Frenet coordinates.")
            return
        else:
            xs = np.atleast_1d(x).astype(np.float64)
            ys = np.atleast_1d(y).astype(np.float64)
            s_arr, d_arr = self.converter.get_frenet(xs, ys)  
            s = float(s_arr[0])
            d = float(d_arr[0])
            self.car_s = s

        H_odom_bl = quaternion_matrix([q.x, q.y, q.z, q.w])
        H_odom_bl[0,3] = x
        H_odom_bl[1,3] = y
        H_odom_bl[2,3] = self.car_pose.position.z

        H_map_odom = np.eye(4, dtype=np.float64)

        self.H_map_bl = H_map_odom @ H_odom_bl
        self.t_map_bl = Time.from_msg(self.current_stamp)  # odom.header.stamp

    def path_callback(self, msg: WaypointArray):
        """Initialize track arrays from global_centerline WaypointArray."""

        if msg is None or len(msg.wpnts) == 0:
            self.get_logger().warn("global_centerline is empty.")
            return

        if self.s_array is not None and not self.path_needs_update:
            return

        xs, ys = [], []
        ss_raw, dl_raw, dr_raw = [], [], []

        for wp in msg.wpnts:
            xs.append(float(wp.x_m))
            ys.append(float(wp.y_m))
            ss_raw.append(float(wp.s_m))
            dl_raw.append(float(wp.d_left))
            dr_raw.append(float(wp.d_right))

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        ss_raw = np.array(ss_raw, dtype=np.float64)
        dl_raw = np.array(dl_raw, dtype=np.float64)
        dr_raw = np.array(dr_raw, dtype=np.float64)

        self.waypoints = np.column_stack([xs, ys]).astype(np.float64)
        self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])

        track_length = float(ss_raw[-1])
        self.track_length = track_length

        ds = 0.05 
        s_dense = np.arange(0.0, track_length, ds)

        s_ext = np.concatenate([ss_raw, [ss_raw[0] + track_length]])
        dl_ext = np.concatenate([dl_raw, [dl_raw[0]]])
        dr_ext = np.concatenate([dr_raw, [dr_raw[0]]])

        dl_dense = np.interp(s_dense, s_ext, dl_ext)
        dr_dense = np.interp(s_dense, s_ext, dr_ext)

        self.s_array = s_dense
        self.d_left_array = np.maximum(dl_dense - self.boundary_inflation, 0.0)
        self.d_right_array = np.maximum(dr_dense - self.boundary_inflation, 0.0)

        self.smallest_d = float(np.min(self.d_right_array + self.d_left_array))
        self.biggest_d  = float(np.max(self.d_right_array + self.d_left_array))

        points = []
        for s_i, dl_i, dr_i in zip(self.s_array,
                                   self.d_left_array,
                                   self.d_right_array):
            x_r, y_r = self.converter.get_cartesian(s_i, -dr_i)
            points.append(Point(x=float(x_r), y=float(y_r), z=0.0))

            x_l, y_l = self.converter.get_cartesian(s_i, dl_i)
            points.append(Point(x=float(x_l), y=float(y_l), z=0.0))

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = marker.SPHERE_LIST
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points = points

        self.pub_boundaries.publish(marker)

        self.path_needs_update = False
        self.track_ready = True
        self.get_logger().info(
            f"global_centerline received: {len(msg.wpnts)} pts, "
            f"track_length={self.track_length:.2f}, "
            f"samples={len(self.s_array)}"
        )

    def _H_from_tf(self, tf_msg):
        """Build 4x4 homogeneous transform from geometry_msgs/TransformStamped."""
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        H = quaternion_matrix([q.x, q.y, q.z, q.w])
        H[0, 3], H[1, 3], H[2, 3] = t.x, t.y, t.z
        return H

    def _transform_xy(self, pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply 4x4 homogeneous transform to 2D XY points (assume z=0)."""
        if pts_xy.size == 0:
            return pts_xy
        N = pts_xy.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.hstack([pts_xy, np.zeros((N,1)), ones])    # z=0
        out = (H @ pts_h.T).T
        return out[:, :2].astype(np.float64)

    def _quat_to_yaw(self, q):
        """Extract yaw (Z-rotation) from quaternion."""
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # def _lookup_tf_exact_or_backoff(self, target: str, source: str, t_scan: Time,
    #                                 future_backoff_sec: float = 0.03, timeout_sec: float = 0.7):
    #     """Try TF at scan time; if 'future extrapolation', back off slightly. Do not silently use latest."""
    #     timeout = Duration(seconds=timeout_sec)

    #     # 1) exact scan time
    #     if self.tf_buffer.can_transform(target, source, t_scan, timeout):
    #         return self.tf_buffer.lookup_transform(target, source, t_scan, timeout)

    #     # 2) small backoff for future-extrapolation
    #     bt = Time(nanoseconds=max(0, t_scan.nanoseconds - int(future_backoff_sec * 1e9)))
    #     if self.tf_buffer.can_transform(target, source, bt, timeout):
    #         return self.tf_buffer.lookup_transform(target, source, bt, timeout)

    #     # 3) try latest: if latest exists, accept it as "static-like"
    #     if self.tf_buffer.can_transform(target, source, Time(), timeout):
    #         tf_latest = self.tf_buffer.lookup_transform(target, source, Time(), timeout)
    #         self.get_logger().warn(
    #             f"TF not ready for {target}<-{source} at scan/backoff; using latest as static edge."
    #         )
    #         return tf_latest

    #     # 4) give up
    #     self.get_logger().warn(
    #         f"TF not ready for {target}<-{source} at scan/backoff/latest.\nKnown frames:\n"
    #         + self.tf_buffer.all_frames_as_yaml()
    #     )
    #     return None

    # def _lookup_tf_exact_or_backoff(self, target: str, source: str, t_scan: Time,
    #                             future_backoff_sec: float = 0.03, timeout_sec: float = 0.7):  # 0.03
    #     """Try TF at scan time, else small backoff for future-extrapolation; never use 'latest' silently."""
    #     timeout = Duration(seconds=timeout_sec)
    #     if self.tf_buffer.can_transform(target, source, t_scan, timeout):
    #         return self.tf_buffer.lookup_transform(target, source, t_scan, timeout)
    #     # small backoff (handles future extrapolation)
    #     bt = Time(nanoseconds=max(0, t_scan.nanoseconds - int(future_backoff_sec*1e9)))
    #     if self.tf_buffer.can_transform(target, source, bt, timeout):
    #         return self.tf_buffer.lookup_transform(target, source, bt, timeout)
    #     # give a clear log and let caller decide to drop frame
    #     self.get_logger().warn(f"TF not ready for {target}<-{source} at scan/backoff. ")
    #     #                    f"Known frames:\n{self.tf_buffer.all_frames_as_yaml()}")
    #     return None

    def _lookup_tf_exact_or_backoff(self, target: str, source: str, t_scan: Time,
                                future_backoff_sec: float = 0.08, timeout_sec: float = 0.08):
        """
        Try lookup at scan time with a short timeout; on Extrapolation, back off slightly once.
        Total at most TWO short waits, no can_transform pre-waits.
        """
        timeout = Duration(seconds=timeout_sec)

        # 1) try exact time
        try:
            return self.tf_buffer.lookup_transform(target, source, t_scan, timeout)
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            # future extrapolation → small backoff
            bt = Time(nanoseconds=max(0, t_scan.nanoseconds - int(future_backoff_sec*1e9)))
            try:
                return self.tf_buffer.lookup_transform(target, source, bt, timeout)
            except Exception as ex2:
                self.get_logger().warn(
                    f"TF not ready for {target}<-{source} at scan/backoff "
                    f"(timeout={timeout_sec:.2f}s, backoff={future_backoff_sec:.3f}s)."
                )
                return None

    def clearmarkers(self):
        # Create a DELETEALL marker
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL # Clear all markers
        arr = MarkerArray()
        arr.markers.append(m)
        return arr

    def laser_to_bl_points(self, scan: LaserScan):
        """Transform LaserScan points into ego_racecar/base_link using static extrinsic."""
        if scan is None:
            return None
        N = len(scan.ranges)
        rng = np.asarray(scan.ranges, dtype=np.float64)
        ang = scan.angle_min + np.arange(N, dtype=np.float64)*scan.angle_increment
        valid = np.isfinite(rng)
        rng, ang = rng[valid], ang[valid]
        if rng.size == 0:
            return None

        # points in laser frame
        x_l = rng*np.cos(ang); y_l = rng*np.sin(ang)
        pts_l = np.stack([x_l, y_l], axis=1)

        # static TF: base_link <- laser (latest is fine because it is static)
        # laser_frame = self._normalize_laser_frame(getattr(scan.header, "frame_id", "ego_racecar/laser"))
        laser_frame = 'ego_racecar/laser_model'
        try:
            tf_bl_from_l = self.tf_buffer.lookup_transform("ego_racecar/base_link", laser_frame, Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed base_link<-{laser_frame}: {e}")
            return None

        H_bl_l = self._H_from_tf(tf_bl_from_l)
        pts_bl = self._transform_xy(pts_l, H_bl_l)
        return pts_bl

    def adaptive_breakpoint_clustering(self, points, d_phi):
        """Adaptive Breakpoint clustering (Amin et al., 2022)."""
        clusters = [[points[0]]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i])
            d_max = (dist * math.sin(d_phi) / math.sin(self.lambda_angle - d_phi) + 3 * self.sigma) / 2
            if np.linalg.norm(points[i] - points[i - 1]) > d_max:
                clusters.append([points[i]])
            else:
                clusters[-1].append(points[i])
        clusters = [np.array(c) for c in clusters if len(c) >= self.min_points_per_cluster]
        # [np.array(c) for c in clusters]
        return clusters
    
    def is_track_boundary(self, s, d):
        """Check if the point (s, d) is on the track boundary."""
        ds = normalize_s(s - self.car_s, self.track_length)
        # if ds < 0 or ds > self.max_viewing_distance:
        #     return True
        if normalize_s(s - self.car_s, self.track_length) > self.max_viewing_distance:
            # print("s out of range")
            return True
        idx = bisect_left(self.s_array, s)
        if idx:
            idx -= 1
        if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
            return True
        return False
    
    # def fit_rectangle(self, objects_pointcloud_list):
    #     current_obstacle_array = []
    #     for c in objects_pointcloud_list:
    #         pts = np.array(c)
    #         center = np.mean(pts, axis=0)
    #         size = np.max(np.linalg.norm(pts - center, axis=1)) * 2
    #         size = float(np.clip(size, 0.3, 0.8))
    #         current_obstacle_array.append(Obstacle(center[0], center[1], size, 0.0))
    #     return current_obstacle_array
    
    def fit_rectangle(self, objects_pointcloud_list):    
        current_obstacle_array = []
        min_dist = self.min_2_points_dist
        for obstacle in objects_pointcloud_list:

            # --- fit a rectangle to the data points ---
            theta = np.linspace(0,np.pi/2 - np.pi/180,90)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            distance1 = np.dot(obstacle, [cos_theta, sin_theta])
            distance2 = np.dot(obstacle, [-sin_theta, cos_theta])
            D10 = -distance1 + np.amax(distance1, axis=0)
            D11 = distance1 - np.amin(distance1, axis=0)
            D20 = -distance2 + np.amax(distance2, axis=0)
            D21 = distance2 - np.amin(distance2, axis=0)
            min_array = np.argmin([np.linalg.norm(D10, axis=0), np.linalg.norm(D11, axis=0)], axis=0)
            D10 = np.transpose(D10)
            D11 = np.transpose(D11)
            D10[min_array == 1] = D11[min_array == 1]
            D10 = np.transpose(D10)
            min_array = np.argmin([np.linalg.norm(D20, axis=0), np.linalg.norm(D21,axis=0)], axis=0)
            D20 = np.transpose(D20)
            D21 = np.transpose(D21)
            D20[min_array == 1] = D21[min_array == 1]
            D20 = np.transpose(D20)
            D = np.minimum(D10, D20)
            D[D < min_dist] = min_dist

            # --------------------------------------------
            # extract the center of the obstacle assuming
            # that it is actually a square obstacle
            # --------------------------------------------

            theta_opt = np.argmax(np.sum(np.reciprocal(D), axis=0))*np.pi/180
            distances1 = np.dot(obstacle, [np.cos(theta_opt), np.sin(theta_opt)])
            distances2 = np.dot(obstacle, [-np.sin(theta_opt), np.cos(theta_opt)])
            max_dist1 = np.max(distances1)
            min_dist1 = np.min(distances1)
            max_dist2 = np.max(distances2)
            min_dist2 = np.min(distances2)

            # corners are detected in a anti_clockwise manner
            corner1 = None
            corner2 = None
            if(np.var(distances2) > np.var(distances1)): # the obstacle has more detection in the verticle direction
                if (np.linalg.norm(-distances1+max_dist1) < np.linalg.norm(distances1-min_dist1)):
                    # the detections are nearer to the right edge
                    # lower_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1 - np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1 + np.cos(theta_opt)*min_dist2])
                    # upper_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1 - np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1 + np.cos(theta_opt)*max_dist2])
                else :
                    # the detections are nearer to the left edge
                    # upper_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1 - np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1 + np.cos(theta_opt)*max_dist2])
                    # lower_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1 - np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1 + np.cos(theta_opt)*min_dist2])
            else: # the obstacle has more detection in the horizontal direction
                if (np.linalg.norm(-distances2 + max_dist2)<np.linalg.norm(distances2 - min_dist2)):
                    # the detections are nearer to the top edge
                    # upper_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1 - np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1 + np.cos(theta_opt)*max_dist2])
                    # upper_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1 - np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1 + np.cos(theta_opt)*max_dist2])
                else :
                    # the detections are nearer to the bottom edge
                    # lower_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1 - np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1 + np.cos(theta_opt)*min_dist2])
                    # lower_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1 - np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1 + np.cos(theta_opt)*min_dist2])
            # vector that goes from corner1 to corner2
            colVec = np.array([corner2[0] - corner1[0], corner2[1] - corner1[1]])
            # orthogonal vector to the one that goes from corner1 to corner2
            orthVec = np.array([-colVec[1], colVec[0]])
            # center position
            center = corner1 + 0.5*colVec + 0.5*orthVec

            current_obstacle_array.append(Obstacle(center[0], center[1], np.linalg.norm(colVec), theta_opt))
            # # center position: 只用可见边的中点，不再加 orthVec
            # center_edge = corner1 + 0.5 * colVec

            # # 可选：沿着“从车指向障碍”的方向稍微外推一点，近似真实车中心
            # # 这里假设激光在 ego_racecar/base_link 原点，cluster 是“近边”
            # vec_to_edge = center_edge
            # dist_edge = np.linalg.norm(vec_to_edge)
            # if dist_edge > 1e-6:
            #     dir_radial = vec_to_edge / dist_edge
            # else:
            #     dir_radial = np.array([1.0, 0.0])  # fallback

            # half_width = 0.25  # 对方车“半宽”（m），可以按实际车宽调
            # center = center_edge + half_width * dir_radial

            # # 大小：用可见边长度作为 size，并且做个夹紧，防止异常跳变
            # raw_size = np.linalg.norm(colVec)
            # size = float(np.clip(raw_size, 0.3, 0.5))  # 下限/上限可按比赛车尺寸调

            # current_obstacle_array.append(
            #     Obstacle(center[0], center[1], size, theta_opt)
            # )


        return current_obstacle_array
    
    def publish_obstacles_message(self):
        obstacles_array_message = ObstacleArray()
        obstacles_array_message.header.stamp = self.current_stamp
        obstacles_array_message.header.frame_id = "map"  

        x_center = []
        y_center = []
        for obstacle in self.tracked_obstacles:
            x_center.append(obstacle.center_x)
            y_center.append(obstacle.center_y)

        s_points, d_points = self.converter.get_frenet(np.array(x_center), np.array(y_center))

        for idx, obstacle in enumerate(self.tracked_obstacles):
            s = s_points[idx]
            d = d_points[idx]

            obsMsg = ObstacleMessage()
            obsMsg.id = obstacle.id
            obsMsg.s_start = s - obstacle.size/2
            obsMsg.s_end = s + obstacle.size/2
            obsMsg.d_left = d + obstacle.size/2
            obsMsg.d_right = d - obstacle.size/2
            obsMsg.s_center = s
            obsMsg.d_center = d
            obsMsg.size = obstacle.size

            obstacles_array_message.obstacles.append(obsMsg)
        self.pub_obstacles_message.publish(obstacles_array_message)

    def publish_markers(self):
        arr = MarkerArray()
        # new_ids = set()

        for i, obs in enumerate(self.tracked_obstacles):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.current_stamp
            m.ns = "obstacles"
            # m.id = int(getattr(obs, "id", i))
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(obs.center_x)
            m.pose.position.y = float(obs.center_y)
            m.pose.position.z = 0.1  
            q = quaternion_from_euler(0.0, 0.0, float(obs.theta))
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            size = float(obs.size)
            m.scale.x = size
            m.scale.y = size
            m.scale.z = max(0.02, size * 0.2)  
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.lifetime = DurationMsg(sec=0, nanosec=int(0.05 * 1e9))

            arr.markers.append(m)
        #     new_ids.add(i)

        # vanished_ids = self.prev_ids - new_ids
        # self.get_logger().info(f"Vanished IDs: {vanished_ids}")
        # for vid in vanished_ids:
        #     m = Marker()
        #     m.header.frame_id = "map"
        #     m.header.stamp = self.current_stamp
        #     m.ns = "obstacles"
        #     m.id = vid
        #     m.action = Marker.DELETE
            # arr.markers.append(m)

        # self.prev_ids = new_ids
        self.pub_markers.publish(self.clearmarkers()) 
        self.pub_markers.publish(arr)

        # Obstacle.current_id = 0

    def publish_obstacles(self, xy, sd):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        # new_ids = set()

        for i in range(len(xy)):
            x, y = xy[i]  
            s, d = sd[i] 

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'map' # "ego_racecar/base_link" 
            m.ns = "obstacles_mid"
            m.id = i
            m.action = Marker.ADD
            m.type = Marker.SPHERE
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.a = 0.9; m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.2
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.orientation.w = 1.0
            m.lifetime = DurationMsg(sec=0, nanosec=int(0.05 * 1e9))
            arr.markers.append(m)
            # new_ids.add(i)

        # vanished_ids = self.prev_ids_obs - new_ids
        # self.get_logger().info(f"Vanished IDs (mid): {vanished_ids}")
        # for vid in vanished_ids:
        #     m = Marker()
        #     m.header.frame_id = "map"
        #     m.header.stamp = stamp
        #     m.ns = "obstacles_mid"
        #     m.id = vid
        #     m.action = Marker.DELETE
        #     arr.markers.append(m)

        # self.prev_ids_obs = new_ids
        # self.pub_breakpoints_markers.publish(self.clearmarkers())
        # self.pub_breakpoints_markers.publish(arr)

    def publish_track_boundaries(self):
        if self.car_s is None: 
            return

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        s0 = self.car_s - 2.0
        s1 = self.car_s + self.max_viewing_distance
        step = 0.5  
        def wrap_s(s): 
            L = self.track_length
            return (s % L + L) % L

        num = int((s1 - s0)/step) + 1
        s_samples = [wrap_s(s0 + k*step) for k in range(num)]

        left_pts = []
        right_pts = []
        for s in s_samples:
            idx = np.searchsorted(self.s_array, s, side='right') - 1
            idx = np.clip(idx, 0, len(self.s_array)-1)
            dL = float(self.d_left_array[idx])
            dR = float(self.d_right_array[idx])

            xyL = self.converter.get_cartesian(s, +dL)
            xyR = self.converter.get_cartesian(s, -dR)
            left_pts.append((float(xyL[0]),  float(xyL[1])))
            right_pts.append((float(xyR[0]), float(xyR[1])))

        left_marker = Marker()
        left_marker.header.frame_id = "map"
        left_marker.header.stamp = stamp
        left_marker.ns = "track_left"
        left_marker.id = 10001
        left_marker.action = Marker.ADD
        left_marker.type = Marker.LINE_STRIP
        left_marker.scale.x = 0.08
        left_marker.color.a = 1.0; left_marker.color.r = 0.1; left_marker.color.g = 0.8; left_marker.color.b = 0.1
        for x, y in left_pts:
            left_marker.points.append(Point(x=float(x), y=float(y), z=0.0))
        arr.markers.append(left_marker)

        right_marker = Marker()
        right_marker.header.frame_id = "map"
        right_marker.header.stamp = stamp
        right_marker.ns = "track_right"
        right_marker.id = 10002
        right_marker.action = Marker.ADD
        right_marker.type = Marker.LINE_STRIP
        right_marker.scale.x = 0.08
        right_marker.color.a = 1.0; right_marker.color.r = 0.1; right_marker.color.g = 0.1; right_marker.color.b = 0.9
        for x, y in right_pts:
            right_marker.points.append(Point(x=float(x), y=float(y), z=0.0))
        arr.markers.append(right_marker)

        self.pub_breakpoints_markers.publish(arr)

    def detect(self):
        """Detect obstacles with clustering in base_link, Frenet check in map, publish in map."""
        if self.converter is None or self.car_s is None or self.scan is None:
            return

        # --- timestamps ---
        scan_t = Time.from_msg(self.scan.header.stamp)
        self.current_stamp = self.scan.header.stamp

        # --- 1) Laser -> base_link (static extrinsic; time-safe) ---
        pts_bl = self.laser_to_bl_points(self.scan)
        if pts_bl is None or pts_bl.shape[0] == 0:
            return

        # --- 2) Cluster in base_link ---
        clusters = self.adaptive_breakpoint_clustering(pts_bl, self.scan.angle_increment)

        # --- 3) For each cluster, compute its midpoint in base_link,
        #         transform that single point to map at scan_t,
        #         compute (s,d) via Frenet, and filter by track boundary ---
        kept_clusters_bl = []
        mids_map = []   # store (x,y) in map for later visualization if desired
        mids_sd   = []  # store (s,d) for later publishing

        MAX_STALENESS_NS = int(0.12 * 1e9)

        use_cached = (self.H_map_bl is not None and self.t_map_bl is not None
                      and abs(scan_t.nanoseconds - self.t_map_bl.nanoseconds) <= MAX_STALENESS_NS) 
        if use_cached:
            # self.get_logger().info('using cached H_map_bl')
            H_map_bl = self.H_map_bl
            yaw_map_from_bl = self._quat_to_yaw(self.car_pose.orientation)
        else:
            # TF: map <- base_link
            tf_map_from_bl = self._lookup_tf_exact_or_backoff("map", "ego_racecar/base_link", scan_t,
                                                              future_backoff_sec=0.1, timeout_sec=0.08)
            if tf_map_from_bl is None:
                self.get_logger().warn('returning from tf_map_from_bl is None')
                return
            self.get_logger().info('tf_map_from_bl is found')
            H_map_bl = self._H_from_tf(tf_map_from_bl)
            yaw_map_from_bl = self._quat_to_yaw(tf_map_from_bl.transform.rotation)

        # for c in clusters:
        #     if c.shape[0] < self.min_obs_size:
        #         continue
        #     mid_bl = c[c.shape[0]//2]               # midpoint in base_link
        #     # dists = np.linalg.norm(c, axis=1)
        #     # mid_bl = c[np.argmin(dists)]   # 用 cluster 最靠近 ego 的点

        #     mid_map = self._transform_xy(mid_bl.reshape(1,2), H_map_bl)[0]  # 1x2 -> 2
        #     # Frenet from map XY
        #     s_arr, d_arr = self.converter.get_frenet(
        #         np.atleast_1d(mid_map[0]).astype(np.float64),
        #         np.atleast_1d(mid_map[1]).astype(np.float64)
        #     )
        #     s, d = float(s_arr[0]), float(d_arr[0])

        #     if self.is_track_boundary(s, d):
        #         continue

        #     kept_clusters_bl.append(c)
        #     mids_map.append((mid_map[0], mid_map[1]))
        #     mids_sd.append((s, d))

        for c in clusters:
            # 1) 太小的簇直接跳过
            if c.shape[0] < self.min_obs_size:
                continue

            # ====== A. 在 cluster 内找“最靠近中心线的点” ======
            # 先把整簇从 base_link 变换到 map
            pts_map_c = self._transform_xy(c, H_map_bl)      # shape (M, 2)
            x_c = pts_map_c[:, 0].astype(np.float64)
            y_c = pts_map_c[:, 1].astype(np.float64)

            # Frenet 投影
            s_c, d_c = self.converter.get_frenet(x_c, y_c)   # numpy arrays
            s_c = s_c.astype(np.float64)
            d_c = d_c.astype(np.float64)

            # 找到 cluster 中 |d| 最小的点 => 最靠近中心线的点
            idx_best = int(np.argmax(np.abs(d_c)))
            s_best = float(s_c[idx_best])
            d_best = float(d_c[idx_best])
            best_map = pts_map_c[idx_best]                   # (x, y) in map

            # ====== B. 用这个点做边界判断 ======
            # 如果连“靠内侧”的点都落在边界上/外面，那整簇当成墙扔掉
            if self.is_track_boundary(s_best, d_best):
                continue

            # ====== C. 这个簇不是边界，保留 ======
            kept_clusters_bl.append(c)

            # 这里用于可视化 / debug：
            # 你可以选择用 best_map / (s_best, d_best)，
            # 也可以继续用几何中心 / mid 点，这里我用 best 的，更一致。
            mids_map.append((best_map[0], best_map[1]))
            mids_sd.append((s_best, d_best))

            # 如果你更希望继续用“簇的中点”作为可视化位置，可以换成：
            # mid_bl = c[c.shape[0] // 2]
            # mid_map = self._transform_xy(mid_bl.reshape(1, 2), H_map_bl)[0]
            # mids_map.append((mid_map[0], mid_map[1]))
            # mids_sd.append((s_best, d_best))  # s,d 仍然用 best 的


        # --- 4) Fit rectangles in base_link (stable geometry) ---
        rects_bl = self.fit_rectangle(kept_clusters_bl)

        # --- 5) Transform rectangle centers/yaws to map at the same scan_t and publish ---
        # yaw_map_from_bl = self._quat_to_yaw(tf_map_from_bl.transform.rotation)
        current_obstacles = []
        for r in rects_bl:
            if r.size > self.max_obs_size:
                continue
            c_bl = np.array([[r.center_x, r.center_y]], dtype=np.float64)
            c_map = self._transform_xy(c_bl, H_map_bl)[0]
            theta_map = float(r.theta + yaw_map_from_bl)
            current_obstacles.append(Obstacle(float(c_map[0]), float(c_map[1]), float(r.size), theta_map))

        # self.tracked_obstacles.clear()
        self.tracked_obstacles = []
        for i, ob in enumerate(current_obstacles):
            ob.id = i
            self.tracked_obstacles.append(ob)
        # self.publish_track_boundaries()
        self.publish_markers()
        # Optional debug of midpoints:
        self.publish_obstacles(mids_map, mids_sd)
        self.publish_obstacles_message()


def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    # node.detect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
