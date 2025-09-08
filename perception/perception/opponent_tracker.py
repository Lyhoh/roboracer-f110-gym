from __future__ import annotations
import math
import numpy as np
from typing import Optional, List
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from interfaces.msg import WaypointArray, ObstacleArray, Obstacle
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter as EKF
from perception.frenet_converter import FrenetConverter

# ---------- small helpers ----------

def normalize_s(s: float, track_length: float) -> float:
    """Wrap s to [0, track_length)."""
    if track_length is None or track_length <= 0:
        return s
    s = s % track_length
    return s

def wrap_s_residual(delta_s: float, track_length: float) -> float:
    """Shortest signed residual on the ring."""
    if track_length is None or track_length <= 0:
        return delta_s
    # Map to (-L/2, L/2]
    delta_s = (delta_s + 0.5 * track_length) % track_length - 0.5 * track_length
    return delta_s

# ---------- tracker core ----------

class SingleOpponentKF:
    """
    EKF in Frenet space with state: [s, vs, d, vd]^T
    - Process: CV (constant velocity) for both s and d
    - Control: optional soft pull toward path: d -> 0, vd -> 0, and vs -> v_target(s)
    - Measurement: [s_meas, vs_meas, d_meas, vd_meas]
    """
    def __init__(self, rate_hz: float,
                 q_vs: float, q_vd: float,
                 r_s: float, r_vs: float, r_d: float, r_vd: float,
                 P0_diag = (0.5, 1.0, 0.2, 0.5)):
        self.rate = max(1.0, float(rate_hz))
        self.dt = 1.0 / self.rate

        # EKF
        self.ekf = EKF(dim_x=4, dim_z=4)  # IMPORTANT: dim_z = 4
        self.ekf.x = np.zeros(4)  # [s, vs, d, vd]

        # State transition (linear CV)
        self.ekf.F = np.array([
            [1.0, self.dt, 0.0,     0.0],
            [0.0, 1.0,     0.0,     0.0],
            [0.0, 0.0,     1.0,     self.dt],
            [0.0, 0.0,     0.0,     1.0]
        ], dtype=float)

        # Process noise (block diag for s-chain and d-chain)
        q1 = Q_discrete_white_noise(dim=2, dt=self.dt, var=max(1e-6, q_vs))
        q2 = Q_discrete_white_noise(dim=2, dt=self.dt, var=max(1e-6, q_vd))
        self.ekf.Q = np.block([
            [q1,              np.zeros((2,2))],
            [np.zeros((2,2)), q2]
        ])

        # Measurement model: direct observe [s, vs, d, vd]
        self.ekf.H = np.eye(4)
        self.R = np.diag([r_s, r_vs, r_d, r_vd])
        self.ekf.R = self.R.copy()

        # Initial covariance
        self.ekf.P = np.diag(P0_diag)

        # Soft-pull gains (can be zero)
        self.P_vs = 0.0
        self.P_d  = 0.0
        self.P_vd = 0.0

        # Track properties
        self.track_length: Optional[float] = None
        self.use_target_vel: bool = False
        self.ratio_to_path: float = 0.6

        # Path reference (for target velocity)
        self.global_waypoints: Optional[List] = None  
        self.wp_step_inv: float = 10.0  # index = int(s * wp_step_inv) like original

        # Simple smoothing for vs/vd outputs
        self.vs_hist = []
        self.vd_hist = []
        self.smooth_len = 5

    # ---- functions for EKF nonlinear interface ----
    def hx(self, x):
        """Measurement function: wraps s to ring, passes others."""
        s, vs, d, vd = x
        s = normalize_s(s, self.track_length) if self.track_length else s
        return np.array([s, vs, d, vd], dtype=float)

    def Hjac(self, x):
        """Jacobian of hx: identity (wrapping is piecewise-constant except discontinuity)."""
        return np.eye(4)

    def residual(self, a, b):
        """Residual in measurement space with ring for s."""
        y = np.array(a) - np.array(b)
        y[0] = wrap_s_residual(y[0], self.track_length if self.track_length else 0.0)
        return y

    # ---- utilities ----
    def set_soft_pulls(self, P_vs: float, P_d: float, P_vd: float):
        self.P_vs = max(0.0, float(P_vs))
        self.P_d  = max(0.0, float(P_d))
        self.P_vd = max(0.0, float(P_vd))

    def set_path(self, wpnts: List, track_length: float, ratio: float = 0.6):
        self.global_waypoints = wpnts
        self.track_length = float(track_length)
        self.ratio_to_path = float(ratio)

    def _v_target_from_path(self, s_val: float) -> float:
        """Look up target longitudinal speed from global path (very lightweight)."""
        if not self.global_waypoints or not self.track_length:
            return 0.0
        # emulate ForzaETH indexing trick
        idx = int((s_val * self.wp_step_inv) % len(self.global_waypoints))
        vx = getattr(self.global_waypoints[idx], 'vx_mps', 0.0)
        return float(self.ratio_to_path * vx)

    # ---- main steps ----
    def initialize_from_two(self, s2: float, s1: float, d2: float, d1: float):
        """Initialize state using two consecutive detections."""
        vs = (wrap_s_residual(s2 - s1, self.track_length) * self.rate)
        vd = (d2 - d1) * self.rate
        x0 = np.array([normalize_s(s2, self.track_length), vs, d2, vd], dtype=float)
        self.ekf.x = x0

    def predict(self):
        """EKF prediction with optional soft pull control."""
        s, vs, d, vd = self.ekf.x
        if self.use_target_vel:
            v_tgt = self._v_target_from_path(s)
            # Control vector u = [0, P_vs*(v_tgt - vs), -P_d*d, -P_vd*vd]
            u = np.array([0.0, self.P_vs * (v_tgt - vs), -self.P_d * d, -self.P_vd * vd], dtype=float)
        else:
            u = np.array([0.0, 0.0, -self.P_d * d, -self.P_vd * vd], dtype=float)

        # We model control as B = I (soft pulls directly add on state delta)
        self.ekf.B = np.eye(4)
        self.ekf.predict(u=u)

        # Keep s on the ring
        self.ekf.x[0] = normalize_s(self.ekf.x[0], self.track_length)

    def update(self, s_meas: float, d_meas: float,
               s_prev: Optional[float], d_prev: Optional[float]):
        """Build 4D measurement using current and (optionally) previous point to estimate vs, vd."""
        # estimate instantaneous velocities (weighted two-tap)
        if s_prev is not None:
            vs1 = wrap_s_residual(s_meas - s_prev, self.track_length) * self.rate
        else:
            vs1 = 0.0
        vd1 = (d_meas - (d_prev if d_prev is not None else d_meas)) * self.rate

        z = np.array([
            normalize_s(s_meas, self.track_length),
            vs1,
            d_meas,
            vd1
        ], dtype=float)

        self.ekf.update(z=z, HJacobian=self.Hjac, Hx=self.hx, residual=self.residual)
        self.ekf.x[0] = normalize_s(self.ekf.x[0], self.track_length)

        # small smoothing buffers
        self.vs_hist.append(self.ekf.x[1])
        self.vd_hist.append(self.ekf.x[3])
        if len(self.vs_hist) > self.smooth_len:
            self.vs_hist.pop(0)
            self.vd_hist.pop(0)

    def get_smoothed_vs(self) -> float:
        return float(np.mean(self.vs_hist)) if self.vs_hist else float(self.ekf.x[1])

    def get_smoothed_vd(self) -> float:
        return float(np.mean(self.vd_hist)) if self.vd_hist else float(self.ekf.x[3])


# ---------- ROS2 node ----------

class OpponentTrackerNode(Node):
    """
    ROS2 node:
    - Subscribes:
        * /perception/detection/raw_obstacles : ObstacleArray (detector output in Frenet)
        * /global_waypoints                   : WaypointsArray     (for track length & target vx)
        * /car_state/odom_frenet              : Odometry      (to know car s for relative ops; optional)
    - Publishes:
        * /perception/obstacles               : ObstacleArray (fused static+dynamic; here we mostly push the dynamic one)
        * /perception/static_dynamic_marker_pub : MarkerArray (RViz markers)
    """
    def __init__(self):
        super().__init__('opponent_tracker')

        # ---- parameters ----
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('P_vs', 0.0)
        self.declare_parameter('P_d',  0.2)
        self.declare_parameter('P_vd', 0.4)

        self.declare_parameter('process_var_vs', 0.3)   # Q for vs chain
        self.declare_parameter('process_var_vd', 0.3)   # Q for vd chain

        self.declare_parameter('meas_var_s',  0.05)     # R diag
        self.declare_parameter('meas_var_vs', 0.8)
        self.declare_parameter('meas_var_d',  0.05)
        self.declare_parameter('meas_var_vd', 0.8)

        self.declare_parameter('ratio_to_path', 0.6)
        self.declare_parameter('use_target_vel_when_lost', True)

        self.declare_parameter('assoc_max_dist_s', 6.0) # gating in Frenet
        self.declare_parameter('assoc_max_dist_d', 1.0)
        self.declare_parameter('ttl_frames', 40)
        self.declare_parameter('var_pub_max', 0.5)      # publish when P[0,0] < var_pub_max

        # ---- read parameters ----
        rate = float(self.get_parameter('rate').value)
        P_vs = float(self.get_parameter('P_vs').value)
        P_d  = float(self.get_parameter('P_d').value)
        P_vd = float(self.get_parameter('P_vd').value)

        q_vs = float(self.get_parameter('process_var_vs').value)
        q_vd = float(self.get_parameter('process_var_vd').value)
        r_s  = float(self.get_parameter('meas_var_s').value)
        r_vs = float(self.get_parameter('meas_var_vs').value)
        r_d  = float(self.get_parameter('meas_var_d').value)
        r_vd = float(self.get_parameter('meas_var_vd').value)

        self.ratio_to_path = float(self.get_parameter('ratio_to_path').value)
        self.use_target_when_lost = bool(self.get_parameter('use_target_vel_when_lost').value)

        self.assoc_max_s = float(self.get_parameter('assoc_max_dist_s').value)
        self.assoc_max_d = float(self.get_parameter('assoc_max_dist_d').value)
        self.ttl_init    = int(self.get_parameter('ttl_frames').value)
        self.var_pub_max = float(self.get_parameter('var_pub_max').value)

        # ---- tracker ----
        self.tracker = SingleOpponentKF(rate, q_vs, q_vd, r_s, r_vs, r_d, r_vd)
        self.tracker.set_soft_pulls(P_vs, P_d, P_vd)

        # ---- state ----
        self.track_length: Optional[float] = None
        self.global_wpnts: Optional[List]  = None
        self.waypoints: Optional[np.ndarray] = None 

        # Simple target container (since single opponent)
        self.has_target = False
        self.target_id = 1
        self.ttl = 0

        # Keep last measurement to estimate vs/vd
        self.prev_s: Optional[float] = None
        self.prev_d: Optional[float] = None

        # ---- QoS ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---- pubs/subs ----
        self.pub_fused  = self.create_publisher(ObstacleArray, '/perception/obstacles', 10)
        self.pub_marker = self.create_publisher(MarkerArray, '/perception/static_dynamic_marker_pub', 10)

        self.sub_obs = self.create_subscription(ObstacleArray, '/opponent_detection/raw_obstacles', self.cb_obstacles, qos)
        self.sub_wp  = self.create_subscription(WaypointArray, '/global_waypoints', self.cb_waypoints, qos)
        # Optional (if you need car s, not required for this simple node)
        # self.sub_car = self.create_subscription(Odometry, '/car_state/odom_frenet',
        #                                         self.cb_car_frenet, qos)

        # ---- timer ----
        self.timer = self.create_timer(1.0 / rate, self.on_timer)

        # self.converter = self.init_frenet_converter()
        self.converter = None

        self.get_logger().info('[OpponentTracker] Node started.')

    # def init_frenet_converter(self):
    #     rospy.wait_for_message("/global_waypoints", WaypointArray)
    #     # Initialize the FrenetConverter object
    #     converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
    #     self.get_logger().info("[OpponentTracker] initialized FrenetConverter object")
    #     return converter

    # --------- callbacks ---------
    def cb_waypoints(self, msg):
        self.global_wpnts = msg.wpnts
        self.waypoints = np.array([[wp.x_m, wp.y_m] for wp in msg.wpnts])
        if len(self.global_wpnts) > 0:
            self.track_length = float(self.global_wpnts[-1].s_m)
            self.tracker.set_path(self.global_wpnts, self.track_length, self.ratio_to_path)
            self.get_logger().info(f'[OpponentTracker] Received global path. Track length = {self.track_length:.2f} m')
        self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])

    def cb_car_frenet(self, msg: Odometry):
        # Not strictly used here, but kept for extension
        pass

    def cb_obstacles(self, msg: ObstacleArray):
        """Associate the single opponent by nearest-neighbor in Frenet."""
        if self.track_length is None or self.track_length <= 0:
            return
        if not msg.obstacles:
            return

        # Choose the measurement closest to current estimate or last meas
        if self.has_target:
            s_ref = float(self.tracker.ekf.x[0])
            d_ref = float(self.tracker.ekf.x[2])
        else:
            # cold start – pick the one with smallest |d| (closest to centerline), fallback to first
            s_ref = float(msg.obstacles[0].s_center)
            d_ref = float(msg.obstacles[0].d_center)

        def dist_sd(obs: Obstacle) -> float:
            ds = wrap_s_residual(float(obs.s_center) - s_ref, self.track_length)
            dd = float(obs.d_center) - d_ref
            # elliptical gate weights
            return (abs(ds) / max(1e-3, self.assoc_max_s)) + (abs(dd) / max(1e-3, self.assoc_max_d))

        best = min(msg.obstacles, key=dist_sd)

        # Gate: ensure it is within allowed window
        ds_gate = abs(wrap_s_residual(best.s_center - s_ref, self.track_length))
        dd_gate = abs(best.d_center - d_ref)
        if self.has_target and (ds_gate > self.assoc_max_s or dd_gate > self.assoc_max_d):
            # No valid association this frame
            return

        # Update last measurement buffer
        s_meas = float(best.s_center)
        d_meas = float(best.d_center)

        if not self.has_target:
            # Need at least two hits to initialize with a velocity
            if self.prev_s is None:
                self.prev_s, self.prev_d = s_meas, d_meas
                return
            # Initialize EKF
            self.tracker.initialize_from_two(s_meas, self.prev_s, d_meas, self.prev_d)
            self.has_target = True
            self.ttl = self.ttl_init
            self.prev_s, self.prev_d = s_meas, d_meas
            return

        # Normal EKF update path
        self.tracker.update(s_meas, d_meas, self.prev_s, self.prev_d)
        self.prev_s, self.prev_d = s_meas, d_meas
        self.ttl = self.ttl_init  # refresh TTL when seen

    # --------- main loop ---------
    def on_timer(self):
        if self.track_length is None or self.track_length <= 0:
            return

        if self.has_target:
            self.tracker.predict()
            self.ttl -= 1
            if self.ttl <= 0:
                # Lost target
                self.has_target = False
                self.prev_s = None
                self.prev_d = None
                # When lost, optionally keep predicting with path speed
                self.tracker.use_target_vel = bool(self.use_target_when_lost)
        else:
            # idle: still predict softly to keep d stabilized if desired
            self.tracker.predict()

        # Publish outputs
        self.publish_marker()
        self.publish_obstacle()

    # --------- publishers ---------
    def publish_marker(self):
        ma = MarkerArray()

        # Clear call
        clr = Marker()
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        m = Marker()
        m.header.frame_id = 'map'  # adjust to your world frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 1
        m.type = Marker.SPHERE
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.5
        m.color.a = 0.8
        m.color.r = 1.0 if self.has_target else 1.0
        m.color.g = 0.0 if self.has_target else 0.3
        m.color.b = 0.0 if self.has_target else 0.8

        # Convert Frenet (s,d) to map (x,y) only if you have a converter.
        # Here we just place a sphere along s on x-axis as a placeholder.
        # Replace this with your Frenet->Cartesian converter.
        s = float(self.tracker.ekf.x[0])
        d = float(self.tracker.ekf.x[2])
        m.pose.position.x, m.pose.position.y = self.converter.get_cartesian(s, d)
        m.pose.orientation.w = 1.0

        ma.markers.append(m)
        self.pub_marker.publish(ma)

    def publish_obstacle(self):
        # Only publish if covariance on s is reasonably bounded
        if self.tracker.ekf.P[0, 0] > self.var_pub_max:
            return
        oa = ObstacleArray()
        oa.header.frame_id = 'map'
        oa.header.stamp = self.get_clock().now().to_msg()

        o = Obstacle()
        o.id = self.target_id
        o.is_static = False
        o.is_actually_a_gap = False
        o.is_visible = self.has_target

        s = float(self.tracker.ekf.x[0])
        d = float(self.tracker.ekf.x[2])
        o.s_center = normalize_s(s, self.track_length) if self.track_length else s
        o.d_center = d

        # size unknown here; fill if your detector provides
        o.size = 0.3

        # simple band for s-interval & d-bounds
        o.s_start = normalize_s(o.s_center - 0.5 * o.size, self.track_length if self.track_length else 0.0)
        o.s_end   = normalize_s(o.s_center + 0.5 * o.size, self.track_length if self.track_length else 0.0)
        o.d_right = o.d_center - 0.5 * o.size
        o.d_left  = o.d_center + 0.5 * o.size

        # velocities (smoothed)
        o.vs = self.tracker.get_smoothed_vs()
        o.vd = self.tracker.get_smoothed_vd()

        # variances
        o.s_var  = float(self.tracker.ekf.P[0, 0])
        o.vs_var = float(self.tracker.ekf.P[1, 1])
        o.d_var  = float(self.tracker.ekf.P[2, 2])
        o.vd_var = float(self.tracker.ekf.P[3, 3])

        oa.obstacles = [o]
        self.pub_fused.publish(oa)


def main():
    rclpy.init()
    node = OpponentTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
