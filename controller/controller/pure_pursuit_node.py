# Dual-vehicle Pure Pursuit controller (ROS 2, rclpy)
# - Both cars share the SAME base path (centerline/raceline)
# - Each car follows a LATERALLY-OFFSET copy of that path: path_offset = base_xy + d * n_hat
# - Ego runs faster than Opponent (either via vx_mps scaling or constant targets)
# CSV header (semicolon ';' + '#' comments): s_m(0); x_m(1); y_m(2); psi_rad(3); kappa(4); vx_mps(5); ax_mps2(6)

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

def load_path_from_csv(path_csv: str, x_col: int, y_col: int, psi_col: int, vx_col: int,
                       delimiter: str = ';', comment: str = '#'):
    """Load base path arrays: path_xy (Nx2), psi (N), vx (N or None)."""
    rows = []
    with open(path_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or (comment and line.startswith(comment)):
                continue
            parts = [p for p in line.split(delimiter) if p != '']
            rows.append(parts)
    arr = np.array(rows, dtype=float)
    x = arr[:, x_col]
    y = arr[:, y_col]
    # psi from column (index 3 in your header); if missing, estimate from gradient
    if 0 <= psi_col < arr.shape[1]:
        psi = arr[:, psi_col]
    else:
        dx = np.gradient(x); dy = np.gradient(y)
        psi = np.arctan2(dy, dx)
    psi = (psi + np.pi) % (2*np.pi) - np.pi
    # vx from column (index 5 in your header)
    vx = None
    if 0 <= vx_col < arr.shape[1]:
        vx = arr[:, vx_col]
        vx = np.maximum(0.0, vx)  # non-negative
    path_xy = np.stack([x, y], axis=1)
    return path_xy, psi, vx

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
        self.path_xy = path_xy
        self.path_vx = path_vx if (path_vx is not None and len(path_vx) == len(path_xy)) else None

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
    """Controls ego & opponent using laterally-offset copies of the same base path."""
    def __init__(self):
        super().__init__('dual_pure_pursuit')

        # Topics (as per your simulator)
        self.odom_topic_ego = self.declare_parameter('odom_topic_ego', '/ego_racecar/odom').get_parameter_value().string_value
        self.odom_topic_opp = self.declare_parameter('odom_topic_opp', '/opp_racecar/odom').get_parameter_value().string_value
        self.drive_topic_ego = self.declare_parameter('drive_topic_ego', '/drive').get_parameter_value().string_value
        self.drive_topic_opp = self.declare_parameter('drive_topic_opp', '/opp_drive').get_parameter_value().string_value

        # Path files (same base path by default)
        self.path_csv_ego = self.declare_parameter('path_csv_ego', '/home/lyh/ros2_ws/src/f110_gym/global-planning/outputs/map5/traj_race_cl.csv').get_parameter_value().string_value
        self.path_csv_opp = self.declare_parameter('path_csv_opp', '').get_parameter_value().string_value
        self.use_same_path = self.declare_parameter('use_same_path', True).get_parameter_value().bool_value

        # Lateral offsets (meters): + is left of tangent
        self.lat_offset_ego = float(self.declare_parameter('lateral_offset_ego',  0.40).value)
        self.lat_offset_opp = float(self.declare_parameter('lateral_offset_opp', -0.40).value)

        # CSV layout (per your header)
        self.csv_delimiter = self.declare_parameter('csv_delimiter', ';').get_parameter_value().string_value
        self.csv_comment   = self.declare_parameter('csv_comment',  '#').get_parameter_value().string_value
        self.x_col  = int(self.declare_parameter('x_col',  1).value)  # x_m
        self.y_col  = int(self.declare_parameter('y_col',  2).value)  # y_m
        self.psi_col= int(self.declare_parameter('psi_col',3).value)  # psi_rad  <-- NOTE: 3
        self.vx_col = int(self.declare_parameter('vx_col', 5).value)  # vx_mps   <-- NOTE: 5

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

        # Load base path and create offset paths
        if not self.path_csv_ego:
            self.get_logger().error('path_csv_ego is empty (base path required).')
        else:
            try:
                base_xy, base_psi, base_vx = load_path_from_csv(
                    self.path_csv_ego, self.x_col, self.y_col, self.psi_col, self.vx_col,
                    delimiter=self.csv_delimiter, comment=self.csv_comment
                )
                ego_xy = offset_path(base_xy, base_psi, self.lat_offset_ego)
                self.ctrl_ego.set_path(ego_xy, base_vx)  # same vx profile index

                self.get_logger().info(
                    f'Loaded base path: {len(base_xy)} pts | offsets: ego={self.lat_offset_ego:.2f}m, opp={self.lat_offset_opp:.2f}m'
                )

                if self.use_same_path or not self.path_csv_opp:
                    opp_xy = offset_path(base_xy, base_psi, self.lat_offset_opp)
                    self.ctrl_opp.set_path(opp_xy, base_vx)
                else:
                    base_xy2, base_psi2, base_vx2 = load_path_from_csv(
                        self.path_csv_opp, self.x_col, self.y_col, self.psi_col, self.vx_col,
                        delimiter=self.csv_delimiter, comment=self.csv_comment
                    )
                    opp_xy = offset_path(base_xy2, base_psi2, self.lat_offset_opp)
                    self.ctrl_opp.set_path(opp_xy, base_vx2)

            except Exception as e:
                self.get_logger().error(f'Failed to load/offset path(s): {e}')

        # ROS I/O
        self.create_subscription(Odometry, self.odom_topic_ego, self.on_odom_ego, 20)
        self.create_subscription(Odometry, self.odom_topic_opp, self.on_odom_opp, 20)
        self.pub_drive_ego = self.create_publisher(AckermannDriveStamped, self.drive_topic_ego, 20)
        self.pub_drive_opp = self.create_publisher(AckermannDriveStamped, self.drive_topic_opp, 20)

        # Start timers
        now_ns = self.get_clock().now().nanoseconds
        self.ctrl_ego.reset_start(now_ns)
        self.ctrl_opp.reset_start(now_ns)
        self.dt = 1.0 / max(1.0, publish_rate)
        self.timer = self.create_timer(self.dt, self.on_timer)

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
