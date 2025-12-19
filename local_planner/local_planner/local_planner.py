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
        self.declare_parameter('back_to_raceline_before', 3.0) 
        self.declare_parameter('back_to_raceline_after', 3.0)  
        self.declare_parameter('d_margin', 0.2)
        self.declare_parameter('avoidance_resolution', 25)
        self.declare_parameter('only_dynamic_obstacles', True)

        # --- Overtake speed rule ---
        self.declare_parameter('ot_min_speed_delta', 0.5)  # m/s, ego must be at least this much faster than opponent
        self.declare_parameter('ot_speed_scale', 1.05)     # slightly increase base raceline speed
        self.declare_parameter('ot_speed_cap', 6.0)        # m/s, safety cap

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
        self.declare_parameter('ot_finish_margin_s', 1.0)     # m, pass opponent by this much before finishing
        self.declare_parameter('ot_lost_timeout_s', 0.5)      # s, allow losing detection briefly
        self.declare_parameter('ot_cooldown_s', 2.0)          # m in s-space, avoid immediate re-trigger

        self.overtake_active = False
        self.ot_target_id = None
        self.ot_side = None
        self.ot_target_s_end = None
        self.ot_last_seen_time = None
        self.ot_cooldown_until_s = None

        # --- Cached opponent info for dropout handling ---
        self.ot_cached_s_start = None
        self.ot_cached_s_end = None
        self.ot_cached_d_left = None
        self.ot_cached_d_right = None
        self.ot_cached_vs = 0.0

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
            '/global_centerline', 
            self.raceline_wp_cb,
            10
        )

        self.create_subscription(
            ObstacleArray,
            '/perception/obstacles',  
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

        # Cooldown: do not start a new overtake immediately after finishing
        if self.ot_cooldown_until_s is not None:
            ds_cd = (self.ot_cooldown_until_s - cur_s) % self.scaled_max_s
            if ds_cd < self.scaled_max_s / 2:
                # still in cooldown window
                self.publish_empty_outputs()
                return
            else:
                self.ot_cooldown_until_s = None

        # 1) Select relevant obstacle (closest ahead, within lookahead)
        considered_obs = []
        for obs in self.obstacles.obstacles:
            if only_dyn and obs.is_static:
                continue
            # obs.s_start is Frenet s of the obstacle
            ds = (obs.s_start - cur_s) % self.scaled_max_s
            if 0.0 <= ds <= lookahead:
                considered_obs.append(obs)

        now_time = self.get_clock().now()
        
        target_obs = None
        use_cached = False 

        # ---------- Guard: no obstacle in lookahead ----------
        if len(considered_obs) == 0:

            if not self.overtake_active:
                self.publish_empty_outputs()
                return

            lost_timeout = float(self.get_parameter('ot_lost_timeout_s').value)
            if self.ot_last_seen_time is not None:
                dt = (now_time - self.ot_last_seen_time).nanoseconds * 1e-9
            else:
                dt = 999.0

            if dt <= lost_timeout:
                # Use cached opponent info to keep publishing OT
                if (self.ot_cached_s_start is None) or (self.ot_cached_s_end is None):
                    # no cache -> cannot continue safely
                    self.overtake_active = False
                    self.publish_empty_outputs()
                    return
                use_cached = True
            else:
                # Lost too long -> abort overtake
                self.overtake_active = False
                self.ot_target_id = None
                self.ot_side = None
                self.ot_target_s_end = None
                self.ot_last_seen_time = None
                self.publish_empty_outputs()
                return

        if not use_cached:
            # 2) Take the closest obstacle (len > 0 guaranteed here)
            considered_obs.sort(key=lambda o: (o.s_start - cur_s) % self.scaled_max_s)

            if not self.overtake_active:
                # Start a new overtake: pick closest ahead
                target_obs = considered_obs[0]
                self.ot_target_id = getattr(target_obs, "id", None)
            else:
                # During overtake: try to keep the same target
                if self.ot_target_id is not None:
                    for o in considered_obs:
                        if getattr(o, "id", None) == self.ot_target_id:
                            target_obs = o
                            break
                if target_obs is None:
                    target_obs = considered_obs[0]

            # Update "last seen" timestamp and stored s_end
            self.ot_last_seen_time = now_time
            self.ot_target_s_end = float(target_obs.s_end)

            # Cache opponent info for dropout handling
            self.ot_cached_s_start = float(target_obs.s_start)
            self.ot_cached_s_end   = float(target_obs.s_end)
            self.ot_cached_d_left  = float(target_obs.d_left)
            self.ot_cached_d_right = float(target_obs.d_right)
            self.ot_cached_vs      = max(0.0, float(getattr(target_obs, "vs", 0.0)))
        else:
            # cached mode: do NOT touch considered_obs[0] or target_obs
            target_obs = None

        # 3) Decide which side to overtake
        #    Use an approximate local region around obstacle center
        if use_cached:
            s_start = self.ot_cached_s_start
            s_end   = self.ot_cached_s_end
            d_left_obs  = self.ot_cached_d_left
            d_right_obs = self.ot_cached_d_right
            opp_vs = float(self.ot_cached_vs)
        else:
            s_start = float(target_obs.s_start)
            s_end   = float(target_obs.s_end)
            d_left_obs  = float(target_obs.d_left)
            d_right_obs = float(target_obs.d_right)
            opp_vs = max(0.0, float(getattr(target_obs, "vs", 0.0)))
            self.ot_cached_vs = opp_vs  # keep fresh

        s_center = 0.5 * (s_start + s_end)
        # Find closest waypoint index to s_center
        wpnts = self.global_waypoints_msg.wpnts
        s_array = np.array([wp.s_m for wp in wpnts])
        idx_center = int(np.argmin(np.abs(s_array - s_center)))
        wp_center = wpnts[idx_center]

        # Convert waypoint widths to wall coordinates in signed d
        d_left_wall = +wp_center.d_left          # left wall > 0
        d_right_wall = -wp_center.d_right        # right wall < 0

        # Available free space on each side
        left_space  = d_left_wall - d_left_obs
        right_space = d_right_obs - d_right_wall
        
        if not self.overtake_active:
            # Decide side only when starting overtake
            if left_space > right_space:
                side = "left"
            else:
                side = "right"
            self.ot_side = side
        else:
            # Keep previous side
            side = self.ot_side if self.ot_side is not None else ("left" if left_space > right_space else "right")
            self.ot_side = side

        if side == "left":
            d_apex = d_left_obs + evasion_dist
            d_apex = max(d_apex, 0.0)
        else:
            d_apex = d_right_obs - evasion_dist
            d_apex = min(d_apex, 0.0)

        self.last_ot_side = side

        # 4) Define avoidance s-interval
        start_avoid = max(cur_s, s_start - back_before)
        end_avoid   = s_end + back_after

        if end_avoid <= start_avoid:
            # Degenerate interval, do nothing
            return
        
        finish_margin = float(self.get_parameter('ot_finish_margin_s').value)

        # check if overtake finished: ego passed opponent end + margin
        finish_s = (self.ot_target_s_end + finish_margin) % self.scaled_max_s
        ds_to_finish = (finish_s - cur_s) % self.scaled_max_s

        passed_finish = ds_to_finish > (self.scaled_max_s / 2)  # finish point is behind ego (mod-safe)

        if self.overtake_active and passed_finish:
            # Finish overtake
            self.overtake_active = False
            self.ot_target_id = None
            self.ot_side = None
            self.ot_target_s_end = None
            self.ot_last_seen_time = None

            # cooldown 
            cooldown_s = float(self.get_parameter('ot_cooldown_s').value)
            self.ot_cooldown_until_s = (cur_s + cooldown_s) % self.scaled_max_s

            self.publish_empty_outputs()
            return

        # If not active yet, we are starting now
        if not self.overtake_active:
            self.overtake_active = True


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

        # 7) Assign a speed profile 

        v_profile = np.interp(
            s_avoid_mod,
            self.race_s_array,
            self.race_v_array
        )

        # --- Enforce ego faster than opponent during overtake ---
        ot_min_speed_delta = float(self.get_parameter('ot_min_speed_delta').value)
        ot_speed_scale     = float(self.get_parameter('ot_speed_scale').value)
        ot_speed_cap       = float(self.get_parameter('ot_speed_cap').value)

        # 1) start from base profile (raceline) and optionally scale it up a bit
        v_profile = v_profile * ot_speed_scale

        # 2) enforce a minimum: ego >= opponent + delta
        v_min = opp_vs + ot_min_speed_delta
        v_profile = np.maximum(v_profile, v_min)

        # 3) cap speed to avoid unsafe acceleration
        v_profile = np.clip(v_profile, 0.0, ot_speed_cap)

        # v_mean = float(np.mean(v_profile)) if len(v_profile) > 0 else 0.0

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

            # Size
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
