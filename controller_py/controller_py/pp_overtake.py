# Dual-vehicle Pure Pursuit controller (ROS 2, rclpy)
# - Opponent: always follows an offset version of the base path from CSV
# - Ego:      follows base path by default, but switches to local planner's
#             OT path (from /planner/avoidance/otwpnts) when available.

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from interfaces.msg import OTWpntArray, WaypointArray

# def load_path_from_csv(path_csv: str, x_col: int, y_col: int, psi_col: int, vx_col: int,
#                        delimiter: str = ';', comment: str = '#'):
#     """Load base path arrays: path_xy (Nx2), psi (N), vx (N or None)."""
#     rows = []
#     with open(path_csv, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or (comment and line.startswith(comment)):
#                 continue
#             parts = [p for p in line.split(delimiter) if p != '']
#             rows.append(parts)
#     arr = np.array(rows, dtype=float)
#     x = arr[:, x_col]
#     y = arr[:, y_col]
#     # psi from column (index 3 in your header); if missing, estimate from gradient
#     if 0 <= psi_col < arr.shape[1]:
#         psi = arr[:, psi_col]
#     else:
#         dx = np.gradient(x); dy = np.gradient(y)
#         psi = np.arctan2(dy, dx)
#     psi = (psi + np.pi) % (2*np.pi) - np.pi
#     # vx from column (index 5 in your header)
#     vx = None
#     if 0 <= vx_col < arr.shape[1]:
#         vx = arr[:, vx_col]
#         vx = np.maximum(0.0, vx)  # non-negative
#     path_xy = np.stack([x, y], axis=1)
#     return path_xy, psi, vx


def offset_path(path_xy: np.ndarray, psi: np.ndarray, d: float) -> np.ndarray:
    """Shift path laterally by d meters along left normal n̂ = [-sin(psi), cos(psi)]."""
    n = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    return path_xy + d * n


class PPController:
    """Pure Pursuit controller state for one car."""
    def __init__(self, L: float, steer_limit_rad: float,
                 ld_min: float, ld_k: float,
                 publish_rate: float,
                 steering_rate_limit: float = 3.0,
                 # speed handling
                 use_path_speed: bool = True,
                 speed_scale: float = 1.0,
                 v_target_const: float = 3.0,
                 v_target_max: float = 4.0,
                 v_ramp_rate: float = 2.0):
        # Vehicle & control params
        self.L = L
        self.steer_limit = steer_limit_rad
        self.ld_min = ld_min
        self.ld_k = ld_k
        self.publish_rate = publish_rate
        self.steering_rate_limit = steering_rate_limit

        # Speed policy params
        self.use_path_speed = use_path_speed
        self.speed_scale = speed_scale
        self.v_target_const = v_target_const
        self.v_target_max = v_target_max
        self.v_ramp_rate = v_ramp_rate

        # Path and runtime state
        self.path_xy = None         # Nx2
        self.path_vx = None         # N or None
        self.pose_ok = False
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0; self.v = 0.0
        self.last_delta = 0.0
        self.start_ns = 0
        self.dt = 1.0 / max(1.0, publish_rate)

    def set_path(self, path_xy: np.ndarray, path_vx: np.ndarray = None):
        """Set current tracking path for this controller."""
        self.path_xy = path_xy
        if path_vx is not None and len(path_vx) == len(path_xy):
            self.path_vx = path_vx
        else:
            self.path_vx = None

    def reset_start(self, now_ns: int):
        self.start_ns = now_ns
        self.last_delta = 0.0

    def nearest_index(self, px: float, py: float) -> int:
        d = self.path_xy - np.array([px, py])
        return int(np.argmin(np.einsum('ij,ij->i', d, d)))

    def step(self, now_ns: int):
        """Compute (steering, speed) with current pose."""
        if self.path_xy is None or not self.pose_ok:
            return None, 0.0

        # Soft-start ramp (global cap)
        t = max(0.0, 1e-9 * (now_ns - self.start_ns))
        v_ramp_cap = self.v_ramp_rate * t  # increases over time

        # Lookahead selection
        i_near = self.nearest_index(self.x, self.y)
        Ld = max(self.ld_min, self.ld_min + self.ld_k * self.v)
        idx = i_near
        dist_acc = 0.0
        N = len(self.path_xy)
        while dist_acc < Ld:
            j = (idx + 1) % N
            dist_acc += float(np.linalg.norm(self.path_xy[j] - self.path_xy[idx]))
            idx = j
            if idx == i_near:
                break
        target = self.path_xy[idx]

        # Target speed policy
        if self.use_path_speed and (self.path_vx is not None):
            v_des = float(self.speed_scale * self.path_vx[idx])
            v_des = min(v_des, self.v_target_max)
        else:
            v_des = min(self.v_target_const, self.v_target_max)
        # Apply global ramp cap at startup to avoid instant full throttle
        v_cmd = min(v_des, v_ramp_cap)

        # Body-frame transform
        dx = target[0] - self.x
        dy = target[1] - self.y
        c = math.cos(self.yaw); s = math.sin(self.yaw)
        x_b =  c * dx + s * dy
        y_b = -s * dx + c * dy
        x_b = max(0.001, x_b)  # avoid divide-by-zero, behind-target

        # Pure Pursuit curvature and steering
        curvature = 2.0 * y_b / (Ld * Ld)
        delta_cmd = math.atan(self.L * curvature)

        # Steering rate limit & magnitude clamp
        max_step = self.steering_rate_limit * self.dt
        delta_cmd = max(min(delta_cmd, self.last_delta + max_step), self.last_delta - max_step)
        delta_cmd = max(-self.steer_limit, min(self.steer_limit, delta_cmd))
        self.last_delta = delta_cmd

        return float(delta_cmd), float(v_cmd)


class DualPurePursuitNode(Node):
    """Controls ego & opponent using laterally-offset copies of the same base path.
       Ego can override its path with a local planner OT path.
    """
    def __init__(self):
        super().__init__('dual_pure_pursuit')

        # Topics
        self.odom_topic_ego = self.declare_parameter('odom_topic_ego', '/ego_racecar/odom').get_parameter_value().string_value
        self.odom_topic_opp = self.declare_parameter('odom_topic_opp', '/opp_racecar/odom').get_parameter_value().string_value
        self.drive_topic_ego = self.declare_parameter('drive_topic_ego', '/drive').get_parameter_value().string_value
        self.drive_topic_opp = self.declare_parameter('drive_topic_opp', '/opp_drive').get_parameter_value().string_value
        
        # self.global_raceline_topic = self.declare_parameter('global_raceline_topic', '/global_raceline').get_parameter_value().string_value
        self.global_raceline_topic = self.declare_parameter('global_raceline_topic', '/global_centerline').get_parameter_value().string_value

        self.ot_topic = self.declare_parameter('ot_topic', '/planner/avoidance/otwpnts').get_parameter_value().string_value

        # Lateral offsets (meters): + is left of tangent
        self.lat_offset_ego = float(self.declare_parameter('lateral_offset_ego',  1.0).value)  # 0.4
        self.lat_offset_opp = float(self.declare_parameter('lateral_offset_opp', -1.0).value) # -0.4

        # Vehicle & controller params
        L = float(self.declare_parameter('wheelbase', 0.33).value)
        steer_limit_deg = float(self.declare_parameter('steer_limit_deg', 24.0).value)
        steer_limit = math.radians(steer_limit_deg)
        publish_rate = float(self.declare_parameter('publish_rate', 40.0).value)
        steering_rate_limit = float(self.declare_parameter('steering_rate_limit', 3.0).value)

        # Speed policies: Ego faster than Opp
        self.ctrl_ego = PPController(
            L=L, steer_limit_rad=steer_limit,
            ld_min=float(self.declare_parameter('ld_min_ego', 0.6).value),
            ld_k=float(self.declare_parameter('ld_k_ego',   0.3).value),
            publish_rate=publish_rate,
            steering_rate_limit=steering_rate_limit,
            use_path_speed=bool(self.declare_parameter('use_path_speed_ego', True).value),
            speed_scale=float(self.declare_parameter('speed_scale_ego', 1.10).value),  # 10% faster on path vx
            v_target_const=float(self.declare_parameter('v_target_ego', 3.5).value),   # used if not using path vx
            v_target_max=float(self.declare_parameter('v_target_max_ego', 4.5).value),
            v_ramp_rate=float(self.declare_parameter('v_ramp_rate_ego', 2.0).value)
        )
        self.ctrl_opp = PPController(
            L=L, steer_limit_rad=steer_limit,
            ld_min=float(self.declare_parameter('ld_min_opp', 0.6).value),
            ld_k=float(self.declare_parameter('ld_k_opp',   0.3).value),
            publish_rate=publish_rate,
            steering_rate_limit=steering_rate_limit,
            use_path_speed=bool(self.declare_parameter('use_path_speed_opp', True).value),
            speed_scale=float(self.declare_parameter('speed_scale_opp', 0.95).value),  # 5% slower on path vx
            v_target_const=float(self.declare_parameter('v_target_opp', 3.0).value),
            v_target_max=float(self.declare_parameter('v_target_max_opp', 4.0).value),
            v_ramp_rate=float(self.declare_parameter('v_ramp_rate_opp', 2.0).value)
        )

        self.base_xy = None
        self.base_psi = None  
        self.base_vx = None
        self.ego_base_xy = None
        self.ego_using_ot_path = False

        # ROS I/O
        self.create_subscription(Odometry, self.odom_topic_ego, self.on_odom_ego, 20)
        self.create_subscription(Odometry, self.odom_topic_opp, self.on_odom_opp, 20)
        self.pub_drive_ego = self.create_publisher(AckermannDriveStamped, self.drive_topic_ego, 20)
        self.pub_drive_opp = self.create_publisher(AckermannDriveStamped, self.drive_topic_opp, 20)

        self.create_subscription(WaypointArray, self.global_raceline_topic, self.on_global_raceline, 10)
        self.create_subscription(OTWpntArray, self.ot_topic, self.on_ot_path, 10)

        # Start timers
        now_ns = self.get_clock().now().nanoseconds
        self.ctrl_ego.reset_start(now_ns)
        self.ctrl_opp.reset_start(now_ns)
        self.dt = 1.0 / max(1.0, publish_rate)
        self.timer = self.create_timer(self.dt, self.on_timer)

    # ---------------------------------------------------
    # Callbacks
    # ---------------------------------------------------
    def on_global_raceline(self, msg: WaypointArray):
        """
        Receive global raceline (WaypointArray) and build base path + offset paths.
        """
        if len(msg.wpnts) == 0:
            self.get_logger().warn("[DualPP] Global raceline is empty.")
            return

        xs = []
        ys = []
        psis = []
        vxs = []
        for wp in msg.wpnts:
            xs.append(wp.x_m)
            ys.append(wp.y_m)
            psis.append(wp.psi_rad)
            vxs.append(max(0.0, wp.vx_mps))

        base_xy = np.stack([xs, ys], axis=1)
        base_psi = np.array(psis, dtype=float)
        base_vx  = np.array(vxs,  dtype=float)

        self.base_xy = base_xy
        self.base_psi = base_psi
        self.base_vx = base_vx

        ego_xy = offset_path(base_xy, base_psi, self.lat_offset_ego)
        opp_xy = offset_path(base_xy, base_psi, self.lat_offset_opp)

        self.ego_base_xy = ego_xy
        self.ctrl_ego.set_path(ego_xy, base_vx)
        self.ctrl_opp.set_path(opp_xy, base_vx)

        self.get_logger().info(
            f"[DualPP] Received global raceline: {len(base_xy)} points | offsets: ego={self.lat_offset_ego:.2f}m, opp={self.lat_offset_opp:.2f}m"
        )

    def on_ot_path(self, msg: OTWpntArray):
        """
        Callback for local planner's overtaking path.
        If msg.wpnts is non-empty -> ego uses this path.
        If empty -> ego falls back to its base (offset) raceline.
        """
        if len(msg.wpnts) == 0:
            # Fallback to base ego path
            if self.ego_base_xy is not None and not self.ego_using_ot_path:
                return
            if self.ego_base_xy is not None:
                self.ctrl_ego.set_path(self.ego_base_xy, self.base_vx)
                self.ego_using_ot_path = False
                self.get_logger().info("[DualPP] OT path empty -> Ego back to base path.")
            return

        ot_xy = np.array([[wp.x_m, wp.y_m] for wp in msg.wpnts], dtype=float)

        # 这里可以选择是否给 path_vx: 目前简单用 v_target_const，不用 path_vx
        self.ctrl_ego.set_path(ot_xy, None)
        self.ego_using_ot_path = True
        self.get_logger().info(f"[DualPP] Received OT path with {len(ot_xy)} points -> Ego following OT path")

    def on_odom_ego(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.ctrl_ego.x = p.x; self.ctrl_ego.y = p.y; self.ctrl_ego.yaw = yaw
        self.ctrl_ego.v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.ctrl_ego.pose_ok = True

    def on_odom_opp(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.ctrl_opp.x = p.x; self.ctrl_opp.y = p.y; self.ctrl_opp.yaw = yaw
        self.ctrl_opp.v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.ctrl_opp.pose_ok = True

    def on_timer(self):
        now_ns = self.get_clock().now().nanoseconds

        d1, v1 = self.ctrl_ego.step(now_ns)
        if d1 is not None:
            m1 = AckermannDriveStamped()
            m1.drive.steering_angle = float(d1)
            m1.drive.speed = float(v1)
            self.pub_drive_ego.publish(m1)

        d2, v2 = self.ctrl_opp.step(now_ns)
        if d2 is not None:
            m2 = AckermannDriveStamped()
            m2.drive.steering_angle = float(d2)
            m2.drive.speed = float(v2)
            self.pub_drive_opp.publish(m2)


def main():
    rclpy.init()
    node = DualPurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
