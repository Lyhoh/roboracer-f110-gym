import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from roboracer_interfaces.msg import WaypointArray, ObstacleArray, Obstacle as ObstacleMessage
import math
from bisect import bisect_left
from roboracer_utils.frenet_converter import FrenetConverter
from tf_transformations import quaternion_matrix, quaternion_from_euler
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
        self.declare_parameter('max_viewing_distance', 9.0)  #9
        self.declare_parameter('lambda_angle', 5.0 * math.pi / 180.0)  # 5 degrees
        self.declare_parameter('sigma', 0.01)  # standard deviation for adaptive clustering
        self.declare_parameter('min_obs_size', 5)
        self.declare_parameter('min_2_points_dist', 0.1)  # minimum distance between two points to be considered an obstacle
        self.declare_parameter('max_obs_size', 0.8)   # 10

        # === Static wall map (from static_map.npz) ===
        self.declare_parameter('use_static_map', False)
        self.declare_parameter('static_map_path', '/home/lyh/ros2_ws/src/f110_gym/localization/static_map/static_map.npz')
        self.declare_parameter('static_tol', 0.2)  # tolerance in d when matching wall

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

        self.use_static_map = self.get_parameter('use_static_map').get_parameter_value().bool_value
        self.static_map_path = self.get_parameter('static_map_path').get_parameter_value().string_value
        self.static_tol = self.get_parameter('static_tol').get_parameter_value().double_value

        self.static_s_axis = None   # 1D array of s
        self.static_d_left = None   # 1D array of d_left(s)
        self.static_d_right = None  # 1D array of d_right(s)

        if self.use_static_map:
            self._load_static_map(self.static_map_path)

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
        self.pub_static_walls = self.create_publisher(MarkerArray, '/perception/static_walls', 10)

        self.pub_frenet_debug = self.create_publisher(MarkerArray, "/perception/frenet_debug", 1)


        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        wp_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.laser_callback, sensor_qos)    # 10
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.create_subscription(WaypointArray, '/global_centerline', self.path_callback, wp_qos)

        # --- LiDAR raw points visualization (RViz Marker) ---
        self.declare_parameter('viz_lidar_points', True)
        self.declare_parameter('viz_lidar_stride', 2)     # 1=all points, 2=every 2 points, 5=every 5 points
        self.declare_parameter('viz_lidar_z', 0.05)       # slightly above ground

        self.viz_lidar_points = self.get_parameter('viz_lidar_points').get_parameter_value().bool_value
        self.viz_lidar_stride = self.get_parameter('viz_lidar_stride').get_parameter_value().integer_value
        self.viz_lidar_z = self.get_parameter('viz_lidar_z').get_parameter_value().double_value

        self.pub_lidar_points = self.create_publisher(Marker, '/perception/lidar_points', 10)

        # --- Cluster visualization ---
        self.declare_parameter('viz_clusters', True)
        self.declare_parameter('viz_cluster_max', 30)        # limit to avoid RViz overload
        self.declare_parameter('viz_cluster_stride', 1)      # downsample points within each cluster
        self.declare_parameter('viz_cluster_z', 0.08)
        self.declare_parameter('viz_text_z', 0.25)

        self.viz_clusters = self.get_parameter('viz_clusters').get_parameter_value().bool_value
        self.viz_cluster_max = self.get_parameter('viz_cluster_max').get_parameter_value().integer_value
        self.viz_cluster_stride = self.get_parameter('viz_cluster_stride').get_parameter_value().integer_value
        self.viz_cluster_z = self.get_parameter('viz_cluster_z').get_parameter_value().double_value
        self.viz_text_z = self.get_parameter('viz_text_z').get_parameter_value().double_value

        self.pub_cluster_markers = self.create_publisher(MarkerArray, "/perception/cluster_debug", 10)
        # --- Filtered (pre-rect) visualization ---
        self.declare_parameter('viz_filtered_clusters', True)
        self.declare_parameter('viz_filtered_stride', 2)
        self.declare_parameter('viz_filtered_z', 0.10)

        self.viz_filtered_clusters = self.get_parameter('viz_filtered_clusters').get_parameter_value().bool_value
        self.viz_filtered_stride = self.get_parameter('viz_filtered_stride').get_parameter_value().integer_value
        self.viz_filtered_z = self.get_parameter('viz_filtered_z').get_parameter_value().double_value

        self.pub_filtered_markers = self.create_publisher(MarkerArray, "/perception/filtered_pre_rect", 10)



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
        self.boundary_inflation = 0.2 # 0.1, 0.3
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

        # === Debug logging for detected obstacles ===
        self.debug_obs_x = []
        self.debug_obs_y = []
        self.debug_obs_s = []
        self.debug_obs_d = []
        # Debug track boundaries (full left/right walls in map frame)
        self.debug_track_left_x = None
        self.debug_track_left_y = None
        self.debug_track_right_x = None
        self.debug_track_right_y = None
        # === Debug: clustering-level (s,d) used for wall decision ===
        self.debug_cls_s = []      # all s_best
        self.debug_cls_d = []      # all d_best
        self.debug_cls_kept = []   # 1=kept as obstacle, 0=filtered as wall
        self.debug_cls_x = []      # best_map.x in map frame
        self.debug_cls_y = []      # best_map.y in map frame
        # debug: mapping for each kept cluster
        self.debug_map_center_x = []
        self.debug_map_center_y = []
        self.debug_map_geomwall_x = []
        self.debug_map_geomwall_y = []
        self.debug_map_staticwall_x = []
        self.debug_map_staticwall_y = []


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

    def debug_frenet_consistency(self, xs, ys):
        """Check consistency of converter.get_frenet & get_cartesian on centerline."""
        if self.converter is None:
            self.get_logger().warn("converter is None, cannot debug Frenet.")
            return

        max_d = 0.0
        max_pos_err = 0.0

        N = len(xs)
        step = max(1, N // 20)  # sample about 20 points

        for i in range(0, N, step):
            x_i = float(xs[i])
            y_i = float(ys[i])

            s_arr, d_arr = self.converter.get_frenet(
                np.atleast_1d(x_i), np.atleast_1d(y_i)
            )
            s_i = float(s_arr[0])
            d_i = float(d_arr[0])

            x_back, y_back = self.converter.get_cartesian(s_i, d_i)

            pos_err = math.hypot(x_back - x_i, y_back - y_i)
            max_d = max(max_d, abs(d_i))
            max_pos_err = max(max_pos_err, pos_err)

            self.get_logger().info(
                f"[FRENET_DBG] i={i}: s={s_i:.2f}, d={d_i:.4f}, "
                f"pos_err={pos_err:.4f}"
            )

        self.get_logger().info(
            f"[FRENET_DBG] max |d| on centerline = {max_d:.4f}, "
            f"max pos_err = {max_pos_err:.4f}"
        )

    def debug_s_axis(self, xs, ys, ss_raw):
        N = len(xs)
        step = max(1, N // 20)

        for i in range(0, N, step):
            x_i = float(xs[i])
            y_i = float(ys[i])
            s_msg = float(ss_raw[i])

            s_arr, d_arr = self.converter.get_frenet(
                np.atleast_1d(x_i), np.atleast_1d(y_i)
            )
            s_conv = float(s_arr[0])
            d_conv = float(d_arr[0])

            self.get_logger().info(
                f"[S_AXIS] i={i}: s_msg={s_msg:.2f}, s_conv={s_conv:.2f}, "
                f"diff={s_conv - s_msg:.2f}, d_conv={d_conv:.4f}"
            )
    def publish_frenet_normals(self, xs, ys):
        if self.converter is None:
            return

        ma = MarkerArray()
        N = len(xs)
        # step = max(1, N // 30)
        step = 1

        for k, i in enumerate(range(0, N, step)):
            x_i = float(xs[i])
            y_i = float(ys[i])

            # 用 converter 的 get_frenet 取出对应的 s
            s_arr, d_arr = self.converter.get_frenet(
                np.atleast_1d(x_i), np.atleast_1d(y_i)
            )
            s_i = float(s_arr[0])
            self.get_logger().info(f"s_i: {s_i}, d_i: {d_arr[0]}")

            # 在 Frenet 坐标中，取 d=0 和 d=+1.0 再 get_cartesian，连成箭头
            x0, y0 = self.converter.get_cartesian(s_i, 0.0)
            x1, y1 = self.converter.get_cartesian(s_i, 1.0)  # +d 方向

            m = Marker()
            m.header.frame_id = 'map'  # 或 "map"/"odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "frenet_normals"
            m.id = k
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.1   # shaft diameter
            m.scale.y = 0.2   # head diameter
            m.scale.z = 0.2   # head length
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0

            p0 = Point(x=x0, y=y0, z=0.0)
            p1 = Point(x=x1, y=y1, z=0.0)
            m.points = [p0, p1]

            ma.markers.append(m)

        self.pub_frenet_debug.publish(ma)

    def publish_lidar_points_marker(self, pts_map: np.ndarray):
        """
        Publish raw lidar points (already in map frame) as SPHERE_LIST marker.
        pts_map: (N,2) ndarray in 'map' frame
        """
        if pts_map is None or pts_map.shape[0] == 0:
            return

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.current_stamp if self.current_stamp is not None else self.get_clock().now().to_msg()
        m.ns = "lidar_raw"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD

        # sphere size
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.2

        # color 
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0

        # short lifetime so it updates smoothly
        m.lifetime = DurationMsg(sec=0, nanosec=int(0.08 * 1e9))

        z = float(self.viz_lidar_z)
        stride = max(1, int(self.viz_lidar_stride))

        pts = pts_map[::stride]
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in pts]

        self.pub_lidar_points.publish(m)

    def _color_from_id(self, cid: int):
        """Deterministic pseudo-random color from cluster id."""
        # Simple hash -> RGB in [0,1]
        r = ((cid * 37) % 255) / 255.0
        g = ((cid * 67) % 255) / 255.0
        b = ((cid * 97) % 255) / 255.0
        return r, g, b

    def _cluster_bbox_diag(self, pts_xy: np.ndarray) -> float:
        """Return bbox diagonal length for cluster size estimate."""
        if pts_xy is None or pts_xy.shape[0] == 0:
            return 0.0
        xmin, ymin = np.min(pts_xy, axis=0)
        xmax, ymax = np.max(pts_xy, axis=0)
        return float(math.hypot(xmax - xmin, ymax - ymin))
    
    def publish_cluster_debug(self, clusters_bl, H_map_bl: np.ndarray):
        """
        Visualize clusters with:
        - cluster points (SPHERE_LIST, different color)
        - center point (SPHERE)
        - text label showing N and bbox diag
        All published in 'map' frame.
        """
        if not self.viz_clusters:
            return
        if clusters_bl is None or len(clusters_bl) == 0:
            return
        if H_map_bl is None:
            return

        stamp = self.current_stamp if self.current_stamp is not None else self.get_clock().now().to_msg()
        ma = MarkerArray()

        # Optional: clear previous markers (use DELETEALL)
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        max_show = min(len(clusters_bl), max(1, int(self.viz_cluster_max)))
        stride_pts = max(1, int(self.viz_cluster_stride))

        mid_id_base = 30000
        pts_id_base = 31000
        txt_id_base = 32000

        for i in range(max_show):
            c_bl = clusters_bl[i]
            if c_bl is None or c_bl.shape[0] == 0:
                continue

            # Transform whole cluster to map
            c_map = self._transform_xy(c_bl, H_map_bl)

            # Compute center (mean) and size (bbox diagonal)
            center = np.mean(c_map, axis=0)
            diag = self._cluster_bbox_diag(c_map)
            npts = int(c_map.shape[0])

            r, g, b = self._color_from_id(i)

            # (1) Cluster points (SPHERE_LIST)
            mp = Marker()
            mp.header.frame_id = "map"
            mp.header.stamp = stamp
            mp.ns = "cluster_pts"
            mp.id = pts_id_base + i
            mp.type = Marker.SPHERE_LIST
            mp.action = Marker.ADD
            mp.scale.x = 0.2
            mp.scale.y = 0.2
            mp.scale.z = 0.2
            mp.color.a = 1.0
            mp.color.r = r
            mp.color.g = g
            mp.color.b = b
            mp.lifetime = DurationMsg(sec=0, nanosec=int(0.10 * 1e9))
            zc = float(self.viz_cluster_z)

            pts = c_map[::stride_pts]
            mp.points = [Point(x=float(p[0]), y=float(p[1]), z=zc) for p in pts]
            ma.markers.append(mp)

            # # (2) Center point (SPHERE)
            # mc = Marker()
            # mc.header.frame_id = "map"
            # mc.header.stamp = stamp
            # mc.ns = "cluster_center"
            # mc.id = mid_id_base + i
            # mc.type = Marker.SPHERE
            # mc.action = Marker.ADD
            # mc.pose.position.x = float(center[0])
            # mc.pose.position.y = float(center[1])
            # mc.pose.position.z = zc + 0.02
            # mc.pose.orientation.w = 1.0
            # mc.scale.x = 0.12
            # mc.scale.y = 0.12
            # mc.scale.z = 0.12
            # mc.color.a = 1.0
            # mc.color.r = r
            # mc.color.g = g
            # mc.color.b = b
            # mc.lifetime = DurationMsg(sec=0, nanosec=int(0.10 * 1e9))
            # ma.markers.append(mc)

            # (3) Text label: N + diag
            # mt = Marker()
            # mt.header.frame_id = "map"
            # mt.header.stamp = stamp
            # mt.ns = "cluster_text"
            # mt.id = txt_id_base + i
            # mt.type = Marker.TEXT_VIEW_FACING
            # mt.action = Marker.ADD
            # mt.pose.position.x = float(center[0])
            # mt.pose.position.y = float(center[1])
            # mt.pose.position.z = float(self.viz_text_z)
            # mt.pose.orientation.w = 1.0
            # mt.scale.z = 0.22  # text height
            # mt.color.a = 1.0
            # mt.color.r = 1.0
            # mt.color.g = 1.0
            # mt.color.b = 1.0
            # mt.text = f"id={i}  N={npts}  diag={diag:.2f}m"
            # mt.lifetime = DurationMsg(sec=0, nanosec=int(0.10 * 1e9))
            # ma.markers.append(mt)

        self.pub_cluster_markers.publish(ma)

    def publish_filtered_pre_rect(self, kept_clusters_bl, rep_points_map=None):
        """
        Publish filtered clusters BEFORE rectangle fitting.
        - kept_clusters_bl: list of clusters in base_link frame (each is Nx2)
        - rep_points_map: optional list[(x_map,y_map)] of representative points (best_map)
        """
        if not self.viz_filtered_clusters:
            return
        if kept_clusters_bl is None or len(kept_clusters_bl) == 0:
            return

        # Need a valid map<-base_link transform (reuse cached one computed in detect)
        if self.H_map_bl is None:
            return

        stamp = self.current_stamp if self.current_stamp is not None else self.get_clock().now().to_msg()
        ma = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        stride = max(1, int(self.viz_filtered_stride))
        z = float(self.viz_filtered_z)

        # (A) Filtered cluster points (each cluster has its own color)
        for i, c_bl in enumerate(kept_clusters_bl):
            if c_bl is None or c_bl.shape[0] == 0:
                continue

            c_map = self._transform_xy(c_bl, self.H_map_bl)

            r, g, b = self._color_from_id(i)

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "filtered_cluster_pts"
            m.id = 40000 + i
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.lifetime = DurationMsg(sec=0, nanosec=int(0.10 * 1e9))

            pts = c_map[::stride]
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in pts]
            ma.markers.append(m)

        # (B) Representative points (best_map) if provided
        # if rep_points_map is not None and len(rep_points_map) > 0:
        #     mp = Marker()
        #     mp.header.frame_id = "map"
        #     mp.header.stamp = stamp
        #     mp.ns = "filtered_rep_points"
        #     mp.id = 41000
        #     mp.type = Marker.SPHERE_LIST
        #     mp.action = Marker.ADD
        #     mp.scale.x = 0.12
        #     mp.scale.y = 0.12
        #     mp.scale.z = 0.12
        #     mp.color.a = 1.0
        #     mp.color.r = 1.0
        #     mp.color.g = 1.0
        #     mp.color.b = 0.0  # yellow
        #     mp.lifetime = DurationMsg(sec=0, nanosec=int(0.10 * 1e9))

        #     mp.points = [Point(x=float(x), y=float(y), z=z + 0.05) for (x, y) in rep_points_map]
        #     ma.markers.append(mp)

        self.pub_filtered_markers.publish(ma)



    
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
        left_xs, left_ys = [], []
        right_xs, right_ys = [], []

        for s_i, dl_i, dr_i in zip(self.s_array,
                                   self.d_left_array,
                                   self.d_right_array):
            x_r, y_r = self.converter.get_cartesian(s_i, -dr_i)
            right_xs.append(float(x_r))
            right_ys.append(float(y_r))
            points.append(Point(x=float(x_r), y=float(y_r), z=0.0))

            x_l, y_l = self.converter.get_cartesian(s_i, dl_i)
            left_xs.append(float(x_l))
            left_ys.append(float(y_l))
            points.append(Point(x=float(x_l), y=float(y_l), z=0.0))

        # store for debug saving
        self.debug_track_left_x = np.array(left_xs, dtype=np.float64)
        self.debug_track_left_y = np.array(left_ys, dtype=np.float64)
        self.debug_track_right_x = np.array(right_xs, dtype=np.float64)
        self.debug_track_right_y = np.array(right_ys, dtype=np.float64)


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
        if self.use_static_map:
            self.publish_static_walls()

        # self.debug_frenet_consistency(xs, ys)
        # self.debug_s_axis(xs, ys, ss_raw)
        # self.publish_frenet_normals(xs, ys)

    def _load_static_map(self, path: str):
        """Load precomputed static walls (s_axis, d_left, d_right) from npz."""
        try:
            data = np.load(path)
            self.static_s_axis = data["s_axis"].astype(np.float64)
            self.static_d_left = data["d_left"].astype(np.float64)
            self.static_d_right = data["d_right"].astype(np.float64)
            self.get_logger().info(
                f"[static map] Loaded static walls from {path}, "
                f"N={len(self.static_s_axis)}"
            )
        except Exception as e:
            self.get_logger().warn(f"[static map] Failed to load {path}: {e}")
            self.use_static_map = False


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
        # if normalize_s(s - self.car_s, self.track_length) > self.max_viewing_distance:
        #     # print("s out of range")
        #     return True
        idx = bisect_left(self.s_array, s)
        if idx:
            idx -= 1
        # if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
        if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
            # self.get_logger().info(f"Point at s={s:.2f}, d={d:.2f} is out of boundary: d_left={self.d_left_array[idx]:.2f}, d_right={-self.d_right_array[idx]:.2f}")
            return True
        return False
    
    def is_static_background(self, s: float, d: float) -> bool:
        """Check if (s,d) lies on precomputed static wall (left/right)."""
        if (not self.use_static_map or
            self.static_s_axis is None or
            self.static_d_left is None or
            self.static_d_right is None):
            return False

        # wrap s to [0, track_length)
        if self.track_length > 0.0:
            s_wrapped = s % self.track_length
        else:
            s_wrapped = s

        # find nearest index in static_s_axis
        idx = np.searchsorted(self.static_s_axis, s_wrapped) - 1
        idx = np.clip(idx, 0, len(self.static_s_axis) - 1)

        dL = self.static_d_left[idx]
        dR = self.static_d_right[idx]

        tol = self.static_tol

        if not np.isnan(dL) and abs(d - dL) < tol:
            return True
        if not np.isnan(dR) and abs(d - dR) < tol:
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

        # === debug log: save all detected obstacles' (x,y,s,d) ===
        self.debug_obs_x.append(x_center.copy())
        self.debug_obs_y.append(y_center.copy())
        self.debug_obs_s.append(s_points.copy())
        self.debug_obs_d.append(d_points.copy())

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
            m.color.a = 0.7
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
            m.scale.x = m.scale.y = m.scale.z = 0.5
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
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

    def publish_static_walls(self):
        """Visualize static_map walls as line strips in map frame."""
        if (not self.use_static_map or
            self.static_s_axis is None or
            self.static_d_left is None or
            self.static_d_right is None or
            self.converter is None):
            return

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        left_marker = Marker()
        left_marker.header.frame_id = "map"
        left_marker.header.stamp = stamp
        left_marker.ns = "static_walls_left"
        left_marker.id = 20001
        left_marker.type = Marker.LINE_STRIP
        left_marker.action = Marker.ADD
        left_marker.scale.x = 0.05      # 线宽
        left_marker.color.a = 1.0
        left_marker.color.r = 0.1
        left_marker.color.g = 0.9
        left_marker.color.b = 0.1

        right_marker = Marker()
        right_marker.header.frame_id = "map"
        right_marker.header.stamp = stamp
        right_marker.ns = "static_walls_right"
        right_marker.id = 20002
        right_marker.type = Marker.LINE_STRIP
        right_marker.action = Marker.ADD
        right_marker.scale.x = 0.05
        right_marker.color.a = 1.0
        right_marker.color.r = 0.1
        right_marker.color.g = 0.1
        right_marker.color.b = 0.9

        for s, dL, dR in zip(self.static_s_axis,
                             self.static_d_left,
                             self.static_d_right):

            if not np.isnan(dL):
                xL, yL = self.converter.get_cartesian(float(s), float(dL))
                left_marker.points.append(
                    Point(x=float(xL), y=float(yL), z=0.0)
                )

            if not np.isnan(dR):
                xR, yR = self.converter.get_cartesian(float(s), float(dR))
                right_marker.points.append(
                    Point(x=float(xR), y=float(yR), z=0.0)
                )

        arr.markers.append(left_marker)
        arr.markers.append(right_marker)

        self.pub_static_walls.publish(arr)


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
        
        # --- visualize raw lidar points in RViz ---
        if self.viz_lidar_points:
            # Use the same transform logic as detection to get map<-base_link at scan time
            scan_t = Time.from_msg(self.scan.header.stamp)
            MAX_STALENESS_NS = int(0.12 * 1e9)

            use_cached = (self.H_map_bl is not None and self.t_map_bl is not None
                        and abs(scan_t.nanoseconds - self.t_map_bl.nanoseconds) <= MAX_STALENESS_NS)

            if use_cached:
                H_map_bl_viz = self.H_map_bl
            else:
                tf_map_from_bl_viz = self._lookup_tf_exact_or_backoff(
                    "map", "ego_racecar/base_link", scan_t,
                    future_backoff_sec=0.1, timeout_sec=0.08
                )
                H_map_bl_viz = self._H_from_tf(tf_map_from_bl_viz) if tf_map_from_bl_viz is not None else None

            if H_map_bl_viz is not None:
                pts_map = self._transform_xy(pts_bl, H_map_bl_viz)  # (N,2) in map
                self.publish_lidar_points_marker(pts_map)


        # --- 2) Cluster in base_link ---
        clusters = self.adaptive_breakpoint_clustering(pts_bl, self.scan.angle_increment)

        # --- 3) For each cluster, compute its representative point in base_link,
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

        # after you have H_map_bl and yaw_map_from_bl
        if self.viz_clusters:
            self.publish_cluster_debug(clusters, H_map_bl)

        for c in clusters:
            if c.shape[0] < self.min_obs_size:
                continue

            pts_map_c = self._transform_xy(c, H_map_bl)      # shape (M, 2)
            x_c = pts_map_c[:, 0].astype(np.float64)
            y_c = pts_map_c[:, 1].astype(np.float64)

            s_c, d_c = self.converter.get_frenet(x_c, y_c)   # numpy arrays
            s_c = s_c.astype(np.float64)
            d_c = d_c.astype(np.float64)

            # idx_best = int(np.argmax(np.abs(d_c)))
            idx_best = int(np.argmin(np.abs(d_c)))
            s_best = float(s_c[idx_best])
            d_best = float(d_c[idx_best])
            best_map = pts_map_c[idx_best]                   # (x, y) in map

            # self.get_logger().info(f"[DEBUG] Cluster size={c.shape[0]}, best (s,d)=({s_best:.2f}, {d_best:.2f})")
            filtered_as_wall = False

            # --- Static map subtraction: skip clusters lying on precomputed walls ---
            if self.use_static_map:
                if self.is_static_background(s_best, d_best):
                    filtered_as_wall = True
                    
                # --- Track boundary check: skip clusters outside track boundaries ---
                elif self.is_track_boundary(s_best, d_best):
                    filtered_as_wall = True
            else:
                if self.is_track_boundary(s_best, d_best):
                    filtered_as_wall = True
                # pass
                
            # self.debug_cls_s.append(s_best)
            # self.debug_cls_d.append(d_best)
            # self.debug_cls_kept.append(0 if filtered_as_wall else 1)
            # self.debug_cls_x.append(float(best_map[0]))
            # self.debug_cls_y.append(float(best_map[1]))

            if not filtered_as_wall:
                self.debug_cls_s.append(s_best)
                self.debug_cls_d.append(d_best)
                self.debug_cls_x.append(float(best_map[0]))
                self.debug_cls_y.append(float(best_map[1]))
                self.debug_cls_kept.append(1)

                x_center, y_center = self.converter.get_cartesian(s_best, 0.0)
                self.debug_map_center_x.append(float(x_center))
                self.debug_map_center_y.append(float(y_center))

                idx_tr = bisect_left(self.s_array, s_best)
                if idx_tr:
                    idx_tr -= 1
                dL_tr = float(self.d_left_array[idx_tr])
                dR_tr = float(self.d_right_array[idx_tr])

                if d_best >= 0.0:
                    x_geom, y_geom = self.converter.get_cartesian(s_best, dL_tr)
                else:
                    x_geom, y_geom = self.converter.get_cartesian(s_best, -dR_tr)

                self.debug_map_geomwall_x.append(float(x_geom))
                self.debug_map_geomwall_y.append(float(y_geom))

                x_static = float('nan')
                y_static = float('nan')
                if (self.use_static_map and
                    self.static_s_axis is not None and
                    self.static_d_left is not None and
                    self.static_d_right is not None):

                    s_wrapped = s_best % self.track_length
                    idx_st = np.searchsorted(self.static_s_axis, s_wrapped) - 1
                    idx_st = int(np.clip(idx_st, 0, len(self.static_s_axis)-1))

                    dL_st = float(self.static_d_left[idx_st])
                    dR_st = float(self.static_d_right[idx_st])

                    if d_best >= 0.0 and not np.isnan(dL_st):
                        x_static, y_static = self.converter.get_cartesian(self.static_s_axis[idx_st], dL_st)
                    elif d_best < 0.0 and not np.isnan(dR_st):
                        x_static, y_static = self.converter.get_cartesian(self.static_s_axis[idx_st], dR_st)

                self.debug_map_staticwall_x.append(float(x_static))
                self.debug_map_staticwall_y.append(float(y_static))


            if filtered_as_wall:
                continue

            kept_clusters_bl.append(c)

            # 这里用于可视化 / debug：
            # 你可以选择用 best_map / (s_best, d_best)，
            # 也可以继续用几何中心 / mid 点，这里我用 best 的，更一致。
            # mids_map.append((best_map[0], best_map[1]))
            # mids_sd.append((s_best, d_best))

            # 如果你更希望继续用“簇的中点”作为可视化位置，可以换成：
            mid_bl = c[c.shape[0] // 2]
            mid_map = self._transform_xy(mid_bl.reshape(1, 2), H_map_bl)[0]
            mids_map.append((mid_map[0], mid_map[1]))
            mids_sd.append((s_best, d_best))  # s,d 仍然用 best 的

        # --- Publish filtered (pre-rect) result ---
        if self.viz_filtered_clusters:
            # ensure self.H_map_bl points to the H_map_bl used this frame
            self.H_map_bl = H_map_bl
            self.publish_filtered_pre_rect(kept_clusters_bl, rep_points_map=mids_map)


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

    def save_obstacle_debug(self, path_name: str = "obstacles_debug.npz"):
        """Save logged obstacle positions (x,y,s,d) and track to an npz file."""
        # 拼成一维数组：每帧一个小数组 -> concat
        if self.debug_obs_x:
            obs_x = np.concatenate(self.debug_obs_x)
            obs_y = np.concatenate(self.debug_obs_y)
            obs_s = np.concatenate(self.debug_obs_s)
            obs_d = np.concatenate(self.debug_obs_d)
        else:
            obs_x = np.array([], dtype=np.float64)
            obs_y = np.array([], dtype=np.float64)
            obs_s = np.array([], dtype=np.float64)
            obs_d = np.array([], dtype=np.float64)

        # 轨道中心线（如果已知）
        if self.waypoints is not None:
            track_x = self.waypoints[:, 0].astype(np.float64)
            track_y = self.waypoints[:, 1].astype(np.float64)
        else:
            track_x = np.array([], dtype=np.float64)
            track_y = np.array([], dtype=np.float64)

        # track boundaries (may be None if path_callback never ran)
        if self.debug_track_left_x is not None:
            left_x = self.debug_track_left_x
            left_y = self.debug_track_left_y
            right_x = self.debug_track_right_x
            right_y = self.debug_track_right_y
        else:
            left_x = np.array([], dtype=np.float64)
            left_y = np.array([], dtype=np.float64)
            right_x = np.array([], dtype=np.float64)
            right_y = np.array([], dtype=np.float64)

        # cluster-level (s_best,d_best) and kept/filtered flag
        if self.debug_cls_s:
            cls_s = np.array(self.debug_cls_s, dtype=np.float64)
            cls_d = np.array(self.debug_cls_d, dtype=np.float64)
            cls_kept = np.array(self.debug_cls_kept, dtype=np.int8)
            cls_x = np.array(self.debug_cls_x, dtype=np.float64)
            cls_y = np.array(self.debug_cls_y, dtype=np.float64)
        else:
            cls_s = np.array([], dtype=np.float64)
            cls_d = np.array([], dtype=np.float64)
            cls_kept = np.array([], dtype=np.int8)
            cls_x = np.array([], dtype=np.float64)
            cls_y = np.array([], dtype=np.float64)
        
            # mapping points for each cluster
        if self.debug_map_center_x:
            center_x = np.array(self.debug_map_center_x, dtype=np.float64)
            center_y = np.array(self.debug_map_center_y, dtype=np.float64)
            geom_x = np.array(self.debug_map_geomwall_x, dtype=np.float64)
            geom_y = np.array(self.debug_map_geomwall_y, dtype=np.float64)
            st_x = np.array(self.debug_map_staticwall_x, dtype=np.float64)
            st_y = np.array(self.debug_map_staticwall_y, dtype=np.float64)
        else:
            center_x = center_y = geom_x = geom_y = st_x = st_y = np.array([], dtype=np.float64)


        path = '/home/lyh/ros2_ws/src/f110_gym/perception/results/' + path_name
        np.savez(path,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_s=obs_s,
                obs_d=obs_d,
                center_x=center_x, center_y=center_y,
                geom_x=geom_x,   geom_y=geom_y,
                st_x=st_x,       st_y=st_y,
                track_x=track_x,
                track_y=track_y,
                left_x=left_x,
                left_y=left_y,
                right_x=right_x,
                right_y=right_y,
                cls_s=cls_s,
                cls_d=cls_d,
                cls_kept=cls_kept,
                cls_x=cls_x,
                cls_y=cls_y,
                track_length=self.track_length)

        self.get_logger().info(
            f"[debug] saved {obs_x.size} obstacle samples to {path}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, saving obstacle debug...")
        # node.save_obstacle_debug("obstacles_debug.npz")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
