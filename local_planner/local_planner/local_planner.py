import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from roboracer_interfaces.msg import Waypoint, WaypointArray, ObstacleArray, OTWpntArray, Ready
from visualization_msgs.msg import Marker, MarkerArray

from roboracer_utils.frenet_converter import FrenetConverter


class SimpleSQPAvoidanceNode(Node):
    """
    A simplified local avoidance planner for ROS2 + f1tenth_gym.

    Idea:
    - Work in Frenet frame (s, d).
    - When a dynamic obstacle (opponent) is detected ahead,
      generate a smooth lateral offset trajectory d(s)
      that passes around the opponent and then comes back to the centerline.
    - Convert (s, d) back to (x, y) and publish as overtaking waypoints.
    """

    def __init__(self):
        super().__init__('simple_sqp_avoidance_node')

        # --- Parameters ---
        self.declare_parameter('lookahead', 15.0)
        self.declare_parameter('evasion_dist', 0.6)
        self.declare_parameter('back_to_raceline_before', 3.0)  # 3.0
        self.declare_parameter('back_to_raceline_after', 3.0)  # 3.0
        self.declare_parameter('d_margin', 0.2)
        self.declare_parameter('avoidance_resolution', 25)
        self.declare_parameter('only_dynamic_obstacles', True)

        # Internal states
        self.frenet_state = Odometry()
        self.global_waypoints_msg = None
        self.global_waypoints_xy = None  # numpy array of shape (N, 2)
        self.scaled_max_s = None

        self.global_raceline_msg = None
        self.race_s_array = None
        self.race_d_array = None
        self.race_v_array = None

        self.obstacles = ObstacleArray()
        self.converter: FrenetConverter = None
        self.track_ready = False
        self.path_needs_update = True

        # For continuity
        self.last_ot_side = "right"  # "left" or "right"
        self.past_avoidance_d = None

        # Subscriptions
        self.create_subscription(
            Odometry,
            '/ego_frenet',   
            self.ego_frenet_cb,
            10
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            WaypointArray,
            '/global_centerline',  
            self.global_wp_cb,
            qos
        )

        self.create_subscription(
            WaypointArray,
            '/global_centerline', # 'global_raceline'
            self.raceline_wp_cb,
            10
        )

        self.create_subscription(
            ObstacleArray,
            '/perception/obstacles',  # your detection/tracking output
            self.obstacles_cb,
            10
        )

        # Publisher
        self.ot_pub = self.create_publisher(
            OTWpntArray,
            '/planner/avoidance/otwpnts',
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            "/planner/avoidance/path",
            10,
        )
 
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/planner/avoidance/markers",
            10,
        )

        self.pub_ready = self.create_publisher(Ready, "/local_planner/ready", 1)


        # Timer loop
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.get_logger().info("[SimpleSQP] Node initialized.")

    # -------------------- Callbacks -------------------- #

    def ego_frenet_cb(self, msg: Odometry):
        """Callback for ego vehicle Frenet odometry."""
        self.frenet_state = msg

    def global_wp_cb(self, msg: WaypointArray):
        """Callback for global waypoints (centerline)."""
        self.global_waypoints_msg = msg
        # Build an array of (x, y) for FrenetConverter
        xy = np.array([[wp.x_m, wp.y_m] for wp in msg.wpnts], dtype=float)
        self.global_waypoints_xy = xy
        self.scaled_max_s = msg.wpnts[-1].s_m
        # Initialize Frenet converter once
        if self.converter is None:
            self.converter = FrenetConverter(xy[:, 0], xy[:, 1])
            self.get_logger().info("[SimpleSQP] FrenetConverter initialized.")

    def raceline_wp_cb(self, msg: WaypointArray):
        """Callback for global raceline waypoints (in centerline Frenet frame)."""
        if self.track_ready and not self.path_needs_update:
            return  
        
        self.global_raceline_msg = msg

        # Extract arrays for fast interpolation
        self.race_s_array = np.array([wp.s_m for wp in msg.wpnts], dtype=float)
        self.race_d_array = np.array([wp.d_m for wp in msg.wpnts], dtype=float)

        # If vx_mps exists, use it; otherwise default to a constant speed
        v_list = []
        for wp in msg.wpnts:
            if hasattr(wp, 'vx_mps'):
                v_list.append(float(wp.vx_mps))
            else:
                v_list.append(3.0)
        self.race_v_array = np.array(v_list, dtype=float)

        self.track_ready = True
        self.path_needs_update = False
        self.pub_ready.publish(Ready(ready=True))

        self.get_logger().info("[SimpleSQP] Raceline waypoints received.")

    def obstacles_cb(self, msg: ObstacleArray):
        """Callback for perception obstacles."""
        self.obstacles = msg

    # -------------------- Main loop -------------------- #

    def timer_callback(self):
        """Main planning loop."""
        if (self.converter is None or
            self.global_waypoints_msg is None or
            self.global_raceline_msg is None):
            # self.get_logger().warn("[SimpleSQP] Waiting for centerline, raceline and FrenetConverter...")
            return

        # Current Frenet state
        cur_s = self.frenet_state.pose.pose.position.x
        cur_d = self.frenet_state.pose.pose.position.y

        lookahead = float(self.get_parameter('lookahead').value)
        evasion_dist = float(self.get_parameter('evasion_dist').value)
        back_before = float(self.get_parameter('back_to_raceline_before').value)
        back_after = float(self.get_parameter('back_to_raceline_after').value)
        d_margin = float(self.get_parameter('d_margin').value)
        resolution = int(self.get_parameter('avoidance_resolution').value)
        only_dyn = bool(self.get_parameter('only_dynamic_obstacles').value)

        # 1) Select relevant obstacle (closest ahead, within lookahead)
        considered_obs = []
        for obs in self.obstacles.obstacles:
            if only_dyn and obs.is_static:
                continue
            # obs.s_start is Frenet s of the obstacle
            ds = (obs.s_start - cur_s) % self.scaled_max_s
            if 0.0 <= ds <= lookahead:
                considered_obs.append(obs)

        if not considered_obs:
            # No one ahead -> publish empty OT path (meaning: use normal path)
            empty_msg = OTWpntArray()
            empty_msg.header = Header()
            empty_msg.header.stamp = self.get_clock().now().to_msg()
            empty_msg.header.frame_id = "map"
            self.ot_pub.publish(empty_msg)
            return

        # 2) Take the closest obstacle
        considered_obs.sort(key=lambda o: (o.s_start - cur_s) % self.scaled_max_s)
        target_obs = considered_obs[0]

        # 3) Decide which side to overtake
        #    We assume WaypointArray has d_left (>0) and d_right (<0), like ForzaETH
        #    Use an approximate local region around obstacle center
        s_center = 0.5 * (target_obs.s_start + target_obs.s_end)
        # Find closest waypoint index to s_center
        wpnts = self.global_waypoints_msg.wpnts
        s_array = np.array([wp.s_m for wp in wpnts])
        idx_center = int(np.argmin(np.abs(s_array - s_center)))
        wp_center = wpnts[idx_center]

        # Convert waypoint widths to wall coordinates in signed d
        d_left_wall = +wp_center.d_left          # left wall > 0
        d_right_wall = -wp_center.d_right        # right wall < 0

        # Available free space on each side
        left_space = d_left_wall - target_obs.d_left    # distance between obstacle and left wall
        right_space = target_obs.d_right - d_right_wall # = target_obs.d_right + wp_center.d_right
        # left_space = wp_center.d_left - target_obs.d_left    # free space to the left
        # right_space = target_obs.d_right - wp_center.d_right # free space to the right

        # Choose side with more space
        if left_space > right_space:
            side = "left"
            d_apex = target_obs.d_left + evasion_dist
            if d_apex < 0.0:
                d_apex = 0.0  # ensure apex is on the left side
        else:
            side = "right"
            d_apex = target_obs.d_right - evasion_dist
            if d_apex > 0.0:
                d_apex = 0.0  # ensure apex is on the right side

        self.last_ot_side = side

        # 4) Define avoidance s-interval
        start_avoid = max(cur_s, target_obs.s_start - back_before)
        end_avoid = target_obs.s_end + back_after
        if end_avoid <= start_avoid:
            # Degenerate interval, do nothing
            return

        s_avoid = np.linspace(start_avoid, end_avoid, resolution)
        s_avoid_mod = np.mod(s_avoid, self.scaled_max_s)

        # 5) Build a smooth d(s) profile:
        #    - Start at current d
        #    - Go to d_apex around the center of the interval
        #    - Return to 0 near the end
        d_profile = np.zeros_like(s_avoid_mod)

        # Interpolate raceline lateral offset d_race(s) along s_avoid_mod
        d_race = np.interp(
            s_avoid_mod,
            self.race_s_array,
            self.race_d_array
        )

        # First point: match current d
        d_profile[0] = cur_d

        # Middle: near s_center -> d_apex
        # End: return to 0
        N = len(s_avoid_mod)
        for i in range(1, N):
            ratio = (i / (N - 1))  # from 0 to 1
            if ratio < 0.5:
                # d -> apex
                alpha = 2.0 * ratio   # [0,1]
                d_profile[i] = (1.0 - alpha) * cur_d + alpha * d_apex
            else:
                # apex -> d
                beta = 2 * (ratio - 0.5)
                d_profile[i] = (1.0 - beta) * d_apex + beta * 0.0
                d_profile[i] = (1.0 - beta) * d_apex + beta * d_race[i]

        # 6) Convert (s, d) -> (x, y) using FrenetConverter
        resp = self.converter.get_cartesian(s_avoid_mod, d_profile)
        x_array = resp[0, :]
        y_array = resp[1, :]

        # 7) Assign a speed profile (simple: use global waypoint speed or constant) 
        # # TODO
        # v_profile = []
        # for s_i in s_avoid_mod:
        #     idx = int(np.argmin(np.abs(s_array - s_i)))
        #     v_profile.append(wpnts[idx].vx_mps if hasattr(wpnts[idx], 'vx_mps') else 3.0)
        # v_profile = np.array(v_profile, dtype=float)

        v_profile = np.interp(
            s_avoid_mod,
            self.race_s_array,
            self.race_v_array
        )

        self.publish_ot_waypoints(x_array, y_array, s_avoid_mod, d_profile, v_profile)
        self.publish_path(x_array, y_array)
        self.publish_markers(x_array, y_array, v_profile)


    # -------------------- Publish helpers -------------------- #

    def publish_empty_outputs(self):
        """Publish empty OTWpntArray and clear RViz markers / path."""
        # Empty OT waypoints
        ot_msg = OTWpntArray()
        ot_msg.header.stamp = self.get_clock().now().to_msg()
        ot_msg.header.frame_id = "map"
        self.ot_pub.publish(ot_msg)

        # Empty Path
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        self.path_pub.publish(path)

        # Clear markers
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)

    # def publish_ot_waypoints(self, x_array, y_array, s_array, d_array, v_array):
    #     """Publish OTWpntArray for controller to follow."""
    #     ot_msg = OTWpntArray()
    #     ot_msg.header.stamp = self.get_clock().now().to_msg()
    #     ot_msg.header.frame_id = "map"

    #     wp_list = []
    #     for i, (x, y, s_i, d_i, v_i) in enumerate(
    #         zip(x_array, y_array, s_array, d_array, v_array)
    #     ):
    #         wp = Waypoint()
    #         wp.x_m = float(x)
    #         wp.y_m = float(y)
    #         wp.s_m = float(s_i)
    #         wp.d_m = float(d_i)
    #         wp.vx_mps = float(v_i)
    #         wp_list.append(wp)

    #     ot_msg.wpnts = wp_list
    #     self.ot_pub.publish(ot_msg)

    # def publish_ot_waypoints(self, x_array, y_array, s_array, d_array, v_array):
    #     """Publish OTWpntArray for controller to follow."""
    #     ot_msg = OTWpntArray()
    #     ot_msg.header.stamp = self.get_clock().now().to_msg()
    #     ot_msg.header.frame_id = "map"

    #     wp_list = []
    #     for i, (x, y, s_i) in enumerate(zip(x_array, y_array, s_array)):
    #         wp = Waypoint()  
    #         wp.x_m = float(x)
    #         wp.y_m = float(y)
    #         wp.s_m = float(s_i)

    #         # d_left / d_right 对 OT 路径来说通常没那么重要，可以：
    #         # 方案 A: 简单设为 0
    #         wp.d_left = 0.0
    #         wp.d_right = 0.0

    #         # 方案 B: 从 global_waypoints 用 s 找对应路点，然后拷贝那里的 d_left/d_right
    #         # idx = int(np.argmin(np.abs(global_s_array - s_i)))
    #         # wp.d_left  = self.global_waypoints_msg.wpnts[idx].d_left
    #         # wp.d_right = self.global_waypoints_msg.wpnts[idx].d_right

    #         wp_list.append(wp)

    #     ot_msg.wpnts = wp_list
    #     self.ot_pub.publish(ot_msg)

    def publish_ot_waypoints(self, x_array, y_array, s_array, d_array, v_array):
        """Publish OTWpntArray for controller to follow."""
        ot_msg = OTWpntArray()
        ot_msg.header.stamp = self.get_clock().now().to_msg()
        ot_msg.header.frame_id = "map"

        wp_list = []
        for i, (x, y, s_i, d_i, v_i) in enumerate(
            zip(x_array, y_array, s_array, d_array, v_array)
        ):
            wp = Waypoint()
            wp.id = i
            wp.x_m = float(x)
            wp.y_m = float(y)
            wp.s_m = float(s_i)
            wp.d_m = float(d_i)
            wp.vx_mps = float(v_i)

            # Optionally copy d_left / d_right from centerline at this s
            if self.global_waypoints_msg is not None:
                s_array_center = np.array([w.s_m for w in self.global_waypoints_msg.wpnts])
                idx = int(np.argmin(np.abs(s_array_center - s_i)))
                wp.d_left = self.global_waypoints_msg.wpnts[idx].d_left
                wp.d_right = self.global_waypoints_msg.wpnts[idx].d_right
            else:
                wp.d_left = 0.0
                wp.d_right = 0.0

            wp_list.append(wp)

        ot_msg.wpnts = wp_list
        self.ot_pub.publish(ot_msg)

    def publish_path(self, x_array, y_array):
        """Publish nav_msgs/Path for RViz (line strip)."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        for x, y in zip(x_array, y_array):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0  # no orientation
            path.poses.append(pose)

        self.path_pub.publish(path)

    def publish_markers(self, x_array, y_array, v_array):
        """Publish MarkerArray for RViz (spheres along the avoidance path)."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Simple speed normalization to [0,1] for marker scale
        if len(v_array) > 0:
            vmax = max(1e-3, float(np.max(v_array)))
        else:
            vmax = 1.0

        for i, (x, y, v) in enumerate(zip(x_array, y_array, v_array)):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "avoidance_points"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            # Position
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.05  # small lift from ground
            m.pose.orientation.w = 1.0

            # Size：随速度变化一点点
            scale = 0.1 + 0.2 * (v / vmax)
            m.scale.x = scale
            m.scale.y = scale
            m.scale.z = scale

            m.color.r = 0.1
            m.color.g = 0.7
            m.color.b = 1.0
            m.color.a = 0.9

            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSQPAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
