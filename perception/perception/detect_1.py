import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import WaypointArray
import math
from bisect import bisect_left
import csv
from perception.frenet_converter import FrenetConverter
from tf_transformations import quaternion_matrix, quaternion_from_euler
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
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
        self.declare_parameter('max_obs_size', 5.0)   # 10

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

        self.csv_path = '/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_waypoints.csv'

        # Publishers
        self.pub_markers = self.create_publisher(MarkerArray, '/opponent_detection/markers', 10)   
        self.pub_boundaries = self.create_publisher(Marker, '/opponent_detection/boundaries', 10)
        self.pub_breakpoints_markers = self.create_publisher(MarkerArray, '/opponent_detection/breakpoints', 10)
        # self.pub_debug = self.create_publisher(MarkerArray, '/opponent_detection/track_debug', 10)
        # self.pub_object = self.create_publisher(MarkerArray, '/opponent_detection/object_markers', 10)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)    # 10
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # self.create_subscription(WaypointsMsg, '/global_waypoints', self.path_callback, 10)

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
        self.boundary_inflation = 0.1  

        self.scan = None
        self.tracked_obstacles = []

        # initialize frenet converter
        self.converter = None
        self.path_callback(None) 

    def laser_callback(self, scan):
        # if self.converter is None or self.car_s is None:
        #     self.get_logger().warn("Track not ready or car pose not set, skipping laser callback.")
        #     return
        
        # points_map = self.laser_to_map_points(scan)
        # if points_map is None or len(points_map) == 0:
        #     return
        
        # # Clustering
        # clusters = self.adaptive_breakpoint_clustering(points_map, scan.angle_increment)
        # # clusters = [c for c in clusters if len(c) >= self.min_points_per_cluster]

        # # removing point clouds that are too small or too big or that have their center point not on the track
        # x_points = []
        # y_points = []
        # for obs in clusters:
        #     x_points.append(obs[int(len(obs)/2)][0])
        #     y_points.append(obs[int(len(obs)/2)][1])
        # s_points, d_points = self.converter.get_frenet(np.array(x_points), np.array(y_points))

        # kept_clusters = []
        # kept_xy = []
        # kept_sd = []
        # for idx, obj in enumerate(clusters):
        #     if len(obj) < self.min_obs_size:
        #         continue
        #     # if not (self.is_on_track(s_points[idx], d_points[idx])):
        #     if self.is_track_boundary(s_points[idx], d_points[idx]):
        #         # print(d_points[idx])
        #         # print("Object {} is on the track boundary, skipping.".format(idx))
        #         continue
        #     kept_clusters.append(obj)
        #     kept_xy.append((x_points[idx], y_points[idx]))
        #     kept_sd.append((s_points[idx], d_points[idx]))
        #     # print(len(kept_xy), len(kept_sd))

        # arr = MarkerArray()
        # for idx, obj in enumerate(kept_clusters):
        #     for j, pt in ((0, obj[0]), (2, obj[-1])):
        #         m = Marker()
        #         m.header.frame_id = 'map' # "ego_racecar/base_link"   # "map"
        #         m.header.stamp = self.current_stamp
        #         m.ns = "breakpoints"
        #         m.id = idx*10 + j
        #         m.action = Marker.ADD
        #         m.type = Marker.SPHERE
        #         m.scale.x = m.scale.y = m.scale.z = 0.25
        #         m.color.a = 0.5
        #         m.color.g = 1.0
        #         m.color.r = 0.0
        #         m.color.b = float(idx) / max(1, len(kept_clusters))
        #         m.pose.position.x = float(pt[0])
        #         m.pose.position.y = float(pt[1])
        #         m.pose.orientation.w = 1.0
        #         arr.markers.append(m)

        # # self.pub_breakpoints_markers.publish(self.clearmarkers())
        # # self.pub_breakpoints_markers.publish(arr)

        # self.pub_breakpoints_markers.publish(self.clearmarkers())  
        # self.publish_obstacles(kept_xy, kept_sd)                     
        # self.publish_track_boundaries()       

        # # obstacles = []
        # # for cluster in kept_clusters:
        # #     rect = self.fit_rectangle(cluster)
        # #     if rect is None:
        # #         continue
        # #     obstacles.append(rect)
        # rects = self.fit_rectangle(kept_clusters)

        # current_obstacles = []
        # for obs in rects:
        #     if(obs.size > self.max_obs_size):
        #         continue
        #     current_obstacles.append(obs)

        # self.tracked_obstacles.clear()
        # for idx, curr_obs in enumerate(current_obstacles):
        #     curr_obs.id = idx
        #     self.tracked_obstacles.append(curr_obs)

        # self.publish_markers()
        # print('laser callback')
        self.scan = scan
        # self.detect()

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

        self.detect()

    def path_callback(self, path):
        """Initialize track arrays from global waypoints.
           For now, load from CSV. Later, replace with msg parsing."""
        # print("path callback")
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            cols = {h: i for i, h in enumerate(header)}
            rows = list(reader)     
        
        # if (self.s_array is None or self.path_needs_update) and self.converter is not None:
        if (self.s_array is None or self.path_needs_update) is not None:
            xs, ys, ss, dl, dr = [], [], [], [], []
            points=[]
            self.s_array = []
            self.d_right_array = []
            self.d_left_array = []

            for row in rows:
                xs.append(float(row[cols["x_m"]]))
                ys.append(float(row[cols["y_m"]]))
            self.waypoints = np.column_stack([xs, ys]).astype(np.float64)
            self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
            # for waypoint in path:
            for row in rows:
                if not row:
                    continue
                # xs.append(float(row[cols["x_m"]]))
                # ys.append(float(row[cols["y_m"]]))                 
                ss.append(float(row[cols["s_m"]]))
                dl.append(float(row[cols["d_left"]]))
                dr.append(float(row[cols["d_right"]]))
                resp = self.converter.get_cartesian(float(row[cols["s_m"]]), -float(row[cols["d_right"]]) + self.boundary_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))
                resp = self.converter.get_cartesian(float(row[cols["s_m"]]), float(row[cols["d_left"]]) - self.boundary_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))

            self.s_array = np.array(ss, dtype=np.float64)
            self.d_left_array = np.maximum(np.array(dl, dtype=np.float64) - self.boundary_inflation, 0.0)
            self.d_right_array = np.maximum(np.array(dr, dtype=np.float64) - self.boundary_inflation, 0.0)
            self.smallest_d = min(self.d_right_array + self.d_left_array)
            self.biggest_d = max(self.d_right_array + self.d_left_array)
            self.track_length = float(self.s_array[-1])

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 0
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2  #0.02
            marker.color.a = 1.
            marker.color.g = 0.
            marker.color.r = 1.
            marker.color.b = 0.
            marker.points = points

            self.pub_boundaries.publish(marker)
        self.path_needs_update = False
        self.track_ready = True
    
    def clearmarkers(self) -> MarkerArray:
        # Create a DELETEALL marker
        m = Marker()
        m.action = Marker.DELETEALL # Clear all markers
        arr = MarkerArray()
        arr.markers.append(m)
        return arr
    
    def laser_to_map_points(self, scan):
        """
        Transform laser scan points from laser frame to map frame.
        Args:
            scan: sensor_msgs/LaserScan
        Returns:
            (M,2) numpy array of [x,y] in map frame, or None if TF fails
        """
        # --- Extract angles and ranges ---
        N = len(scan.ranges)
        angles = scan.angle_min + np.arange(N, dtype=np.float64) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=np.float64)
        valid = np.isfinite(ranges)
        angles, ranges = angles[valid], ranges[valid]

        if angles.size == 0:
            return None

        # --- Build homogeneous points in laser frame (z=0) ---
        x_l = ranges * np.cos(angles)
        y_l = ranges * np.sin(angles)
        z_l = np.zeros_like(x_l)
        pts_laser_h = np.vstack((x_l, y_l, z_l, np.ones_like(x_l)))  # shape (4,M)

        # --- Lookup transform map <- laser ---
        laser_frame = 'ego_racecar/laser_model' # scan.header.frame_id
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame="map",
                source_frame=laser_frame,
                time=rclpy.time.Time())  # latest available
        except Exception as e:
            self.get_logger().error(f"TF lookup failed map<-{laser_frame}: {e}")
            return None

        # --- Build homogeneous transform H (map <- laser) ---
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        H = quaternion_matrix([q.x, q.y, q.z, q.w])  # (4,4)
        H[0, 3] = t.x
        H[1, 3] = t.y
        H[2, 3] = t.z

        # --- Transform points to map frame ---
        pts_map_h = H @ pts_laser_h
        points_map = pts_map_h[:2, :].T.astype(np.float64)  # shape (M,2)

        return points_map if len(points_map) > 0 else None

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
        return [np.array(c) for c in clusters]
    
    # def is_on_track(self, s, d):
    #     """Check if the point (s, d) is on the track."""
    #     # print(self.car_s)
    #     if normalize_s(s - self.car_s, self.track_length) > self.max_viewing_distance:
    #         return False
    #     if abs(d) >= self.biggest_d:
    #         return False
    #     if abs(d) <= self.smallest_d:
    #         return True
    #     idx = bisect_left(self.s_array, s)
    #     if idx:
    #         idx -= 1
    #     if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
    #         return False
    #     return True
    
    def is_track_boundary(self, s, d):
        """Check if the point (s, d) is on the track boundary."""
        if normalize_s(s - self.car_s, self.track_length) > self.max_viewing_distance:
            # print("s out of range")
            return True
        idx = bisect_left(self.s_array, s)
        if idx:
            idx -= 1
        if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
            return True
        return False
    
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

        return current_obstacle_array

    # def fit_rectangle(self, cluster):
    #     """Fit an axis-aligned bounding box in car frame."""
    #     x_min, y_min = np.min(cluster, axis=0)
    #     x_max, y_max = np.max(cluster, axis=0)
    #     width = max(1e-3, x_max - x_min)
    #     height = max(1e-3, y_max - y_min)
    #     center_x = (x_max + x_min) / 2
    #     center_y = (y_max + y_min) / 2
    #     return {'center': (center_x, center_y), 'width': width, 'height': height, 'yaw': 0.0}

    # def publish_markers(self):
    #     markers_array = []
    #     for obs in self.tracked_obstacles:
    #         marker = Marker()
    #         marker.header.frame_id = "map"
    #         marker.header.stamp = self.current_stamp
    #         marker.id = obs.id
    #         marker.type = marker.CUBE
    #         marker.scale.x = obs.size
    #         marker.scale.y = obs.size
    #         marker.scale.z = obs.size
    #         marker.color.a = 0.5
    #         marker.color.g = 1.
    #         marker.color.r = 0.
    #         marker.color.b = 1.
    #         marker.pose.position.x = obs.center_x
    #         marker.pose.position.y = obs.center_y
    #         q = quaternion_from_euler(0, 0, obs.theta)
    #         marker.pose.orientation.x = q[0]          # orientation (in quaternion) !
    #         marker.pose.orientation.y = q[1]
    #         marker.pose.orientation.z = q[2]
    #         marker.pose.orientation.w = q[3]
    #         markers_array.append(marker)
    #     self.pub_markers.publish(self.clearmarkers())
    #     self.pub_markers.publish(markers_array)
    #     Obstacle.current_id = 0

    def publish_markers(self):
        arr = MarkerArray()

        for i, obs in enumerate(self.tracked_obstacles):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.current_stamp
            m.ns = "obstacles"
            m.id = int(getattr(obs, "id", i))  
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

            arr.markers.append(m)

        self.pub_markers.publish(self.clearmarkers()) 
        self.pub_markers.publish(arr)
        Obstacle.current_id = 0

        # ---- 1) 发布物体中点 ----
    def publish_obstacles(self, xy, sd):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i in range(len(xy)):
            # 取中点（也可以改成质心：obj.mean(axis=0)）
            x, y = xy[i]  
            s, d = sd[i] 
            # 如果点还在 base_link，需要变到 map
            # mx, my = self.base_link_to_map(cx_bl, cy_bl, self.car_pose)

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
            arr.markers.append(m)

            # # --- 2) 文字标注 s,d ---
            # t = Marker()
            # t.header.frame_id = 'map' # "ego_racecar/base_link"
            # t.header.stamp = stamp
            # t.ns = "obstacles_text"
            # t.id = 1000 + i          # id 必须唯一
            # t.action = Marker.ADD
            # t.type = Marker.TEXT_VIEW_FACING
            # t.scale.z = 0.1          # 字体大小
            # t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            # t.pose.position.x = float(x)
            # t.pose.position.y = float(y)
            # t.pose.position.z = 0.5  # 提高一点避免和小车重叠
            # t.pose.orientation.w = 1.0
            # t.text = f"s={s:.1f},d={d:.1f}"
            # arr.markers.append(t)

        self.pub_breakpoints_markers.publish(arr)

    # ---- 2) 发布赛道边界（只画视距内，效率更高）----
    def publish_track_boundaries(self):
        if self.car_s is None: 
            return

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        # 选取 car_s 附近的一段 s（前视 + 少量后视，用于闭合）
        s0 = self.car_s - 2.0
        s1 = self.car_s + self.max_viewing_distance
        step = 0.5  # 采样间隔(m)，越小越密
        # wrap 到 [0, track_length)
        def wrap_s(s): 
            L = self.track_length
            return (s % L + L) % L

        # 采样 s 列表（考虑环形）
        num = int((s1 - s0)/step) + 1
        s_samples = [wrap_s(s0 + k*step) for k in range(num)]

        # 把每个 s 转为左右边界 xy
        left_pts = []
        right_pts = []
        for s in s_samples:
            # 用 d_left/d_right 查表（与 s_array 对齐）
            idx = np.searchsorted(self.s_array, s, side='right') - 1
            idx = np.clip(idx, 0, len(self.s_array)-1)
            dL = float(self.d_left_array[idx])
            dR = float(self.d_right_array[idx])

            # --- 方法A：有 Frenet->Cartesian 的函数（推荐）
            xyL = self.converter.get_cartesian(s, +dL)
            xyR = self.converter.get_cartesian(s, -dR)
            left_pts.append((float(xyL[0]),  float(xyL[1])))
            right_pts.append((float(xyR[0]), float(xyR[1])))

        # 组装 LINE_STRIP (左/右各一条)
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
        # print('detect')
        if self.converter is None or self.car_s is None:
            self.get_logger().warn("Track not ready or car pose not set, skipping laser callback.")
            return
        
        points_map = self.laser_to_map_points(self.scan)
        if points_map is None or len(points_map) == 0:
            return
        
        # Clustering
        clusters = self.adaptive_breakpoint_clustering(points_map, self.scan.angle_increment)
        # clusters = [c for c in clusters if len(c) >= self.min_points_per_cluster]

        # removing point clouds that are too small or too big or that have their center point not on the track
        x_points = []
        y_points = []
        for obs in clusters:
            x_points.append(obs[int(len(obs)/2)][0])
            y_points.append(obs[int(len(obs)/2)][1])
        s_points, d_points = self.converter.get_frenet(np.array(x_points), np.array(y_points))

        kept_clusters = []
        kept_xy = []
        kept_sd = []
        for idx, obj in enumerate(clusters):
            if len(obj) < self.min_obs_size:
                continue
            # if not (self.is_on_track(s_points[idx], d_points[idx])):
            # if self.is_track_boundary(s_points[idx], d_points[idx]):
            #     # print(d_points[idx])
            #     # print("Object {} is on the track boundary, skipping.".format(idx))
            #     continue
            kept_clusters.append(obj)
            kept_xy.append((x_points[idx], y_points[idx]))
            kept_sd.append((s_points[idx], d_points[idx]))
            # print(len(kept_xy), len(kept_sd))

        arr = MarkerArray()
        for idx, obj in enumerate(kept_clusters):
            for j, pt in ((0, obj[0]), (2, obj[-1])):
                m = Marker()
                m.header.frame_id = 'map' # "ego_racecar/base_link"   # "map"
                m.header.stamp = self.current_stamp
                m.ns = "breakpoints"
                m.id = idx*10 + j
                m.action = Marker.ADD
                m.type = Marker.SPHERE
                m.scale.x = m.scale.y = m.scale.z = 0.25
                m.color.a = 0.5
                m.color.g = 1.0
                m.color.r = 0.0
                m.color.b = float(idx) / max(1, len(kept_clusters))
                m.pose.position.x = float(pt[0])
                m.pose.position.y = float(pt[1])
                m.pose.orientation.w = 1.0
                arr.markers.append(m)

        # self.pub_breakpoints_markers.publish(self.clearmarkers())
        # self.pub_breakpoints_markers.publish(arr)

        self.pub_breakpoints_markers.publish(self.clearmarkers())  
        self.publish_obstacles(kept_xy, kept_sd)                     
        # self.publish_track_boundaries()       

        # obstacles = []
        # for cluster in kept_clusters:
        #     rect = self.fit_rectangle(cluster)
        #     if rect is None:
        #         continue
        #     obstacles.append(rect)
        rects = self.fit_rectangle(kept_clusters)

        current_obstacles = []
        for obs in rects:
            if(obs.size > self.max_obs_size):
                continue
            current_obstacles.append(obs)

        self.tracked_obstacles.clear()
        for idx, curr_obs in enumerate(current_obstacles):
            curr_obs.id = idx
            self.tracked_obstacles.append(curr_obs)

        self.publish_markers()


def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    # node.detect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
