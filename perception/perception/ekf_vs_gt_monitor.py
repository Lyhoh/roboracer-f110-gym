from __future__ import annotations
import math
from collections import deque
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import datetime, os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from roboracer_interfaces.msg import ObstacleArray, WaypointArray
from std_msgs.msg import Float32MultiArray

from roboracer_utils.frenet_converter import FrenetConverter 


def quat_to_yaw(x, y, z, w) -> float:
    """Convert quaternion to yaw (Z)."""
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_s_residual(ds: float, L: float) -> float:
    """Shortest signed residual on a ring of length L."""
    return (ds + 0.5 * L) % L - 0.5 * L if L > 0.0 else ds


def normalize_s(s: float, L: float) -> float:
    """Wrap s into [0, L)."""
    return s % L if L > 0.0 else s


def compute_waypoints_psi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute heading (yaw) angle psi [rad] for each waypoint based on (x, y).
    The yaw is the tangent direction of the path, measured CCW from x-axis.
    """
    # Differentiate with wrap-around 
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)

    psi = np.arctan2(dy, dx)  # atan2 gives angle in [-pi, pi]

    return psi

def compute_curvature_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature kappa along a 2D parametric curve using finite differences:
      kappa = (x' y'' - y' x'') / ( (x'^2 + y'^2)^(3/2) )
    Output has same length as x,y.
    """
    dx  = np.gradient(x, edge_order=2)
    dy  = np.gradient(y, edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)

    denom = (dx*dx + dy*dy)**1.5 + 1e-12
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa

def unwrap_time(t: np.ndarray) -> np.ndarray:
    """Shift time to start at 0 for nicer plots."""
    return t - t[0]

def interp_kappa_at_time(fc, s_true_list, kappa_s, s_wp):
    """
    Build kappa(t) by mapping s(t) -> kappa(s) with interpolation.
    - s_true_list: list/array of s(t) (prefer GT s_true for stability)
    - s_wp: waypoint s grid, must be monotonic over [0, L]
    """
    s_t = np.asarray(s_true_list, dtype=float) % float(fc.raceline_length)
    return np.interp(s_t, s_wp, kappa_s)


class EkfVsGtMonitor(Node):
    def __init__(self, track_name, v_cmd_const):
        super().__init__('ekf_vs_gt_monitor')

        # --- Config (edit if needed) ---
        self.max_time_gap = 0.20   # allowed |t_ekf - t_gt| in seconds
        self.print_every  = 20     # print rolling metrics every N EKF messages
        # -------------------------------

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        qos_wp = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST, depth=1,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.sub_wp  = self.create_subscription(WaypointArray, '/global_centerline', self.cb_wp, qos_wp)
        self.sub_gt  = self.create_subscription(Odometry, '/opp_racecar/odom', self.cb_gt, qos)
        self.sub_ekf = self.create_subscription(ObstacleArray, '/perception/obstacles', self.cb_ekf, qos)

        self.pub_err_cart = self.create_publisher(Float32MultiArray, '/monitor/err_cart', 10)
        self.pub_err_fren = self.create_publisher(Float32MultiArray, '/monitor/err_fren', 10)

        self.fc: Optional[FrenetConverter] = None
        self.track_length: float = 0.0
        self.centerline_xy = None
        
        self.track_ready = False
        self.path_needs_update = True 

        self.gt_buf: deque[Odometry] = deque(maxlen=200)

        # rolling stats for Frenet errors [e_s, e_vs, e_d, e_vd]
        self.n = 0
        self.sum_e  = np.zeros(4)
        self.sum_ea = np.zeros(4)
        self.sum_e2 = np.zeros(4)
        self.maxabs = np.zeros(4)

        # ---- live buffers for plotting (time-aligned to EKF stamps) ----
        self.log_t = []      # seconds
        self.log_es  = []    # e_s
        self.log_evs = []    # e_vs
        self.log_ed  = []    # e_d
        self.log_evd = []    # e_vd
        self.log_ex = []     # Cartesian error X [m]
        self.log_ey = []     # Cartesian error Y [m]
        self.log_dist = []   # Cartesian distance error
        self.log_xy_ref = [] # (x_gt, y_gt)
        self.log_xy_est = [] # (x_est, y_est)
        self.log_s_true = []
        self.log_s_est  = []
        self.log_d_true = []
        self.log_d_est  = []
        self.log_vs_true = []
        self.log_vs_est  = []
        self.log_vd_true = []
        self.log_vd_est  = []

        self.track_name = track_name
        self.v_cmd_const = float(v_cmd_const)
        self.save_dir = "/home/lyh/ros2_ws/src/f110_gym/perception/results/"
        self.save_name_suf = track_name + '_' + str(v_cmd_const).replace('.','_')


        # --- detection (raw) subscription ---
        self.sub_det = self.create_subscription(
            ObstacleArray, '/perception/raw_obstacles', self.cb_det, qos
        )
        self.det_buf: deque[ObstacleArray] = deque(maxlen=200)

        # --- detection vs tracking logging buffers ---
        self.log_xy_det = []        # (x_det, y_det)
        self.log_det_ex = []        # detection - tracking error X
        self.log_det_ey = []        # detection - tracking error Y
        self.log_det_dist = []      # Euclidean |det - trk| in XY

        self.log_det_es = []        # detection - tracking e_s (wrapped)
        self.log_det_ed = []        # detection - tracking e_d

        self.s_wp = None
        self.kappa_s = None

        self.get_logger().info('[EkfVsGtMonitor] node started.')

    # ---------- Waypoints → FrenetConverter ----------
    def cb_wp(self, msg: WaypointArray):
        if self.track_ready and not self.path_needs_update:
            return
        if not msg.wpnts:
            return

        x = np.array([w.x_m for w in msg.wpnts], dtype=float)
        y = np.array([w.y_m for w in msg.wpnts], dtype=float)

        psi = compute_waypoints_psi(x, y)

        self.fc = FrenetConverter(waypoints_x=x, waypoints_y=y, waypoints_psi=psi)
        self.track_length = float(self.fc.raceline_length)
        self.get_logger().info(f'[Monitor] FrenetConverter ready. L={self.track_length:.2f} m.')
        self.track_ready = True
        self.path_needs_update = False

        # --- directly cache the centerline from waypoints ---
        try:
            self.centerline_xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=float)
            self.get_logger().info(f"[Monitor] Cached {len(self.centerline_xy)} centerline points from waypoints.")
        except Exception as e:
            self.centerline_xy = None
            self.get_logger().warn(f"[Monitor] Failed to cache centerline: {e}")

        # 1) Build an s-grid for the cached centerline points (0..L)
        try:
            s_wp = np.array([w.s_m for w in msg.wpnts], dtype=float)
        except Exception:
            pass

        # 2) Curvature along the centerline
        kappa_s = compute_curvature_from_xy(self.centerline_xy[:, 0], self.centerline_xy[:, 1])

        # 3) Store for plotting
        self.s_wp = s_wp
        self.kappa_s = kappa_s



    def cb_det(self, msg: ObstacleArray):
        self.det_buf.append(msg)

    def get_nearest_det(self, t_target: Time) -> Optional[ObstacleArray]:
        if not self.det_buf:
            return None
        best, best_dt = None, 1e9
        for m in reversed(self.det_buf):
            dt = abs((Time.from_msg(m.header.stamp) - t_target).nanoseconds) * 1e-9
            if dt < best_dt:
                best_dt = dt; best = m
            if dt > self.max_time_gap and best is not None:
                break
        return best if best_dt <= self.max_time_gap else None


    # ---------- Buffer GT ----------
    def cb_gt(self, msg: Odometry):
        self.gt_buf.append(msg)

    def get_nearest_gt(self, t_target: Time) -> Optional[Odometry]:
        if not self.gt_buf:
            return None
        best = None
        best_dt = 1e9
        for m in reversed(self.gt_buf):
            dt = abs((Time.from_msg(m.header.stamp) - t_target).nanoseconds) * 1e-9
            if dt < best_dt:
                best_dt = dt
                best = m
            if dt > self.max_time_gap and best is not None:
                break
        return best if best_dt <= self.max_time_gap else None

    # ---------- Main compare ----------
    def cb_ekf(self, ekf_msg: ObstacleArray):
        if self.fc is None or not ekf_msg.obstacles:
            return
        ekf = ekf_msg.obstacles[0]
        t_ekf = Time.from_msg(ekf_msg.header.stamp)
        # self.get_logger().info(f"EKF t={t_ekf.nanoseconds * 1e-9:.3f}s s={ekf.s_center:.2f} d={ekf.d_center:.2f} vs={ekf.vs:.2f} vd={ekf.vd:.2f}")

        gt = self.get_nearest_gt(t_ekf)
        if gt is None:
            return

        # --- EKF estimate in XY using converter (s_est, d_est) → (x_est, y_est)
        s_est = float(ekf.s_center)
        d_est = float(ekf.d_center)
        xy_est = self.fc.get_cartesian(np.array([s_est]), np.array([d_est]))  # returns arrays
        x_est, y_est = float(xy_est[0][0]), float(xy_est[1][0])

        # --- GT pose in XY (map/world)
        x_gt = float(gt.pose.pose.position.x)
        y_gt = float(gt.pose.pose.position.y)

        # Cartesian errors
        ex = x_est - x_gt
        ey = y_est - y_gt
        dist = math.hypot(ex, ey)

        msg_cart = Float32MultiArray()
        msg_cart.data = [float(ex), float(ey), float(dist)]
        # self.pub_err_cart.publish(msg_cart)

        # --- Frenet GT using converter: (x_gt, y_gt) → (s_true, d_true)
        # converter expects arrays; get_frenet returns [s, d] arrays
        sd_true = self.fc.get_frenet(np.array([x_gt]), np.array([y_gt]))
        s_true, d_true = float(sd_true[0][0]), float(sd_true[1][0])

        # GT velocities → (vs_true, vd_true)
        # Option A: if waypoints_psi provided, use converter.get_frenet_velocities
        # Option B: otherwise project (vx,vy) onto tangent/normal using derivative from converter
        vx = float(gt.twist.twist.linear.x)
        vy = float(gt.twist.twist.linear.y)

        vs_true: float
        vd_true: float

        if self.fc.waypoints_psi is not None:
            # need yaw of the vehicle for delta_psi
            q = gt.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            vs_vd = self.fc.get_frenet_velocities(vx=vx, vy=vy, theta=yaw, s=s_true)
            vs_true, vd_true = float(vs_vd[0]), float(vs_vd[1])
        else:
            # project GT velocity onto path tangent/normal at s_true
            dx_ds, dy_ds = self.fc.get_derivative(np.array([s_true]))
            # normalize tangent
            tnorm = math.hypot(float(dx_ds[0]), float(dy_ds[0])) + 1e-12
            tx = float(dx_ds[0]) / tnorm
            ty = float(dy_ds[0]) / tnorm
            nx, ny = -ty, tx
            vs_true = vx * tx + vy * ty
            vd_true = vx * nx + vy * ny

        # EKF velocities from message
        vs_est = float(ekf.vs)
        vd_est = float(ekf.vd)

        # Frenet errors (wrap s on the ring!)
        e_s  = wrap_s_residual(normalize_s(s_est, self.track_length) - normalize_s(s_true, self.track_length),
                               self.track_length)
        e_d  = d_est - d_true
        e_vs = vs_est - vs_true
        e_vd = vd_est - vd_true

        msg_fren = Float32MultiArray()
        msg_fren.data = [float(e_s), float(e_vs), float(e_d), float(e_vd)]
        # self.pub_err_fren.publish(msg_fren)

        # rolling metrics
        self.n += 1
        e = np.array([e_s, e_vs, e_d, e_vd], dtype=float)
        self.sum_e  += e
        self.sum_ea += np.abs(e)
        self.sum_e2 += e**2
        self.maxabs  = np.maximum(self.maxabs, np.abs(e))

        if self.n % self.print_every == 0:
            bias = self.sum_e / self.n
            mae  = self.sum_ea / self.n
            rmse = np.sqrt(self.sum_e2 / self.n)

        # Use EKF stamp as x-axis
        t_ekf_sec = t_ekf.nanoseconds * 1e-9

        self.log_t.append(t_ekf_sec)
        self.log_es.append(e_s)
        self.log_evs.append(e_vs)
        self.log_ed.append(e_d)
        self.log_evd.append(e_vd)
        self.log_ex.append(ex)
        self.log_ey.append(ey)
        self.log_dist.append(dist)
        self.log_xy_ref.append((x_gt, y_gt))
        self.log_xy_est.append((x_est, y_est))
        self.log_s_true.append(s_true);   self.log_s_est.append(s_est)
        self.log_d_true.append(d_true);   self.log_d_est.append(d_est)
        self.log_vs_true.append(vs_true); self.log_vs_est.append(vs_est)
        self.log_vd_true.append(vd_true); self.log_vd_est.append(vd_est)

        # --- nearest detection frame to EKF time ---
        det_arr = self.get_nearest_det(t_ekf)
        det = None
        if det_arr is not None and det_arr.obstacles:
            # pick detection closest to EKF estimate in (s,d)
            s_ref, d_ref = float(ekf.s_center), float(ekf.d_center)
            def sd_cost(o):
                ds = (float(o.s_center) - s_ref + 0.5*self.track_length) % self.track_length - 0.5*self.track_length
                dd = float(o.d_center) - d_ref
                return abs(ds) + abs(dd)
            det = min(det_arr.obstacles, key=sd_cost)

        # --- if we have a detection, compare to EKF (tracking) ---
        if det is not None:
            s_det = float(det.s_center)
            d_det = float(det.d_center)

            # Frenet residuals (det - trk)
            e_s_dt = wrap_s_residual(
                normalize_s(s_det, self.track_length) - normalize_s(s_est, self.track_length),
                self.track_length
            )
            e_d_dt = d_det - d_est

            # Convert detection (s,d) to XY
            xy_det = self.fc.get_cartesian(np.array([s_det]), np.array([d_det]))
            x_det, y_det = float(xy_det[0][0]), float(xy_det[1][0])

            # Cartesian diffs (det - trk)
            ex_dt = x_det - x_est
            ey_dt = y_det - y_est
            dist_dt = math.hypot(ex_dt, ey_dt)

            # Log for later plotting
            self.log_xy_det.append((x_det, y_det))
            self.log_det_ex.append(ex_dt)
            self.log_det_ey.append(ey_dt)
            self.log_det_dist.append(dist_dt)
            self.log_det_es.append(e_s_dt)
            self.log_det_ed.append(e_d_dt)
        else:
            # keep vector lengths aligned if you prefer; or just skip
            pass

    def show_main_3panel(self,
                        spike_percentile: float = 95.0,
                        kappa_percentile: float = 85.0,
                        shade_high_kappa: bool = True):
        """
        Main evaluation figure (3 stacked subplots, shared x):
        1) Cartesian distance error |e_xy|
        2) EKF v_s estimate vs commanded speed (constant)
        3) Track curvature magnitude |kappa|(t) aligned via s_true(t)

        Assumes these buffers exist:
        self.log_t, self.log_dist, self.log_vs_est, self.log_s_true
        and curvature cache:
        self.s_wp, self.kappa_s, self.track_length
        """

        # ---------- Guards ----------
        if len(self.log_t) == 0:
            self.get_logger().warn("No samples collected; cannot plot.")
            return
        if self.s_wp is None or self.kappa_s is None:
            self.get_logger().warn("Curvature not ready (missing waypoints / kappa cache).")
            return

        # ---------- Arrays ----------
        t = np.asarray(self.log_t, dtype=float)
        t = t - t[0]  # start at 0 for readability

        dist = np.asarray(self.log_dist, dtype=float)
        vs_est = np.asarray(self.log_vs_est, dtype=float)
        s_true = np.asarray(self.log_s_true, dtype=float) % float(self.track_length)

        # Downsample for speed (plotting only)
        step = max(1, len(t) // 4000)
        t_p = t[::step]
        dist_p = dist[::step]
        vs_est_p = vs_est[::step]
        s_true_p = s_true[::step]

        # ---------- Curvature aligned to time ----------
        # kappa(s) from waypoints -> kappa(t) via s_true(t)
        kappa_t = np.interp(s_true_p, self.s_wp, self.kappa_s)
        kappa_abs = np.abs(kappa_t)

        # ---------- Spike highlight on distance error ----------
        finite_dist = dist_p[np.isfinite(dist_p)]
        if len(finite_dist) > 0:
            spike_th = float(np.percentile(finite_dist, spike_percentile))
        else:
            spike_th = float("nan")
        spike_mask = np.isfinite(dist_p) & (dist_p >= spike_th)

        # ---------- High-curvature shading (optional) ----------
        finite_k = kappa_abs[np.isfinite(kappa_abs)]
        if len(finite_k) > 0:
            kappa_th = float(np.percentile(finite_k, kappa_percentile))
        else:
            kappa_th = float("nan")
        high_kappa = np.isfinite(kappa_abs) & (kappa_abs >= kappa_th)

        def shade_regions(ax, x, mask, alpha=0.12):
            """Shade contiguous True regions in mask along x."""
            if not np.any(mask):
                return
            idx = np.where(mask)[0]
            # find contiguous blocks
            blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for b in blocks:
                ax.axvspan(x[b[0]], x[b[-1]], alpha=alpha)

        # ---------- Plot ----------
        fig, axes = plt.subplots(3, 1, figsize=(14, 7.8), sharex=True, constrained_layout=True)

        # (1) Cartesian distance error
        ax1 = axes[0]
        ax1.plot(t_p, dist_p, label="Cartesian distance error |e_xy|")
        if np.any(spike_mask):
            ax1.scatter(t_p[spike_mask], dist_p[spike_mask], s=12,
                        label=f"Spikes (≥ P{int(spike_percentile)} = {spike_th:.2f} m)")
        if shade_high_kappa and np.isfinite(kappa_th):
            shade_regions(ax1, t_p, high_kappa, alpha=0.10)
        ax1.set_ylabel("Error [m]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # (2) v_s estimate vs commanded speed (constant)
        ax2 = axes[1]
        ax2.plot(t_p, vs_est_p, label="Estimated v_s (EKF)")
        ax2.plot(t_p, np.full_like(t_p, float(self.v_cmd_const)), linestyle="--",
                label=f"Commanded speed v_cmd = {self.v_cmd_const:.2f} m/s")
        if shade_high_kappa and np.isfinite(kappa_th):
            shade_regions(ax2, t_p, high_kappa, alpha=0.10)
        ax2.set_ylabel("Speed [m/s]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # (3) curvature magnitude
        ax3 = axes[2]
        ax3.plot(t_p, kappa_abs, label="|curvature| |κ(t)|")

        if shade_high_kappa and np.isfinite(kappa_th):
            shade_regions(ax3, t_p, high_kappa, alpha=0.10)
            
        ax3.axhline(kappa_th, linestyle="--",
                    label=f"High-curvature threshold (P{int(kappa_percentile)})")
        ax3.set_ylabel("|κ| [1/m]")
        ax3.set_xlabel("Time [s]")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        # Title (optional)
        fig.suptitle("Perception Evaluation: Position Error, Speed Estimate, and Track Curvature", y=1.02)

        # Save
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, "final_main_3panel_" + self.save_name_suf + ".png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved main 3-panel figure: {out_path}")

        plt.show(block=False)
        plt.pause(0.1)


    def show_hexbin_95ellipse(self):

        ex = np.asarray(self.log_ex, dtype=float)
        ey = np.asarray(self.log_ey, dtype=float)
        mask = np.isfinite(ex) & np.isfinite(ey)
        ex = ex[mask]; ey = ey[mask]
        if len(ex) < 10:
            self.get_logger().warn("Not enough samples for hexbin.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
        hb = ax.hexbin(ex, ey, gridsize=45, cmap="Blues", bins="log", mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("count (log scale)")

        mx, my = float(ex.mean()), float(ey.mean())
        cov = np.cov(np.vstack([ex, ey]))
        vals, vecs = np.linalg.eigh(cov)

        # 95% confidence ellipse for 2D Gaussian:
        # scale = sqrt(chi2.ppf(0.95, df=2)) ≈ 2.4477
        scale = 2.4477

        theta = np.linspace(0, 2*np.pi, 256)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        ell = vecs @ (scale * np.sqrt(vals)[:, None] * circle)

        ax.plot(ell[0] + mx, ell[1] + my, "r--", lw=2.0, label="95% ellipse")
        ax.scatter([mx], [my], marker="x", s=90, color="r", label="mean")

        ax.set_title("Cartesian Error Density (hexbin)")
        ax.set_xlabel("Error X [m]")
        ax.set_ylabel("Error Y [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        fig.savefig(os.path.join(self.save_dir, "final_hexbin_95ellipse_" + self.save_name_suf + ".png"), dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved: {os.path.join(self.save_dir, 'final_hexbin_95ellipse_' + self.save_name_suf + '.png')}")

    def show_track_overlay(self, downsample_max_points: int = 4000):
        """
        Plot GT vs EKF track overlay in XY.
        Uses self.log_xy_ref and self.log_xy_est (already logged in cb_ekf).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if len(self.log_xy_ref) == 0 or len(self.log_xy_est) == 0:
            self.get_logger().warn("No XY logs for track overlay.")
            return

        xy_ref = np.asarray(self.log_xy_ref, dtype=float)
        xy_est = np.asarray(self.log_xy_est, dtype=float)

        n = min(len(xy_ref), len(xy_est))
        xy_ref = xy_ref[:n]
        xy_est = xy_est[:n]

        # Downsample for readability
        step = max(1, n // max(1, downsample_max_points))
        xy_ref_p = xy_ref[::step]
        xy_est_p = xy_est[::step]

        fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)

        ax.plot(xy_ref_p[:, 0], xy_ref_p[:, 1], lw=1.5, label="GT trajectory")
        ax.plot(xy_est_p[:, 0], xy_est_p[:, 1], lw=1.5, label="EKF estimate")

        # plot centerline
        # if getattr(self, "centerline_xy", None) is not None and len(self.centerline_xy) > 1:
        #     cl = np.asarray(self.centerline_xy, dtype=float)
        #     ax.plot(cl[:, 0], cl[:, 1], lw=1.0, alpha=0.6, label="Centerline")

        ax.set_title("Track Overlay (GT vs EKF)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        os.makedirs(self.save_dir, exist_ok=True)
        out_path = os.path.join(self.save_dir, "final_track_overlay_" + self.save_name_suf + ".png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved track overlay: {out_path}")

        plt.show(block=False)
        plt.pause(0.1)

    def compute_kpi_by_speed(self):
        """
        Compute KPIs for a constant commanded speed.
        Outputs metrics suitable for a PPT table.
        """
        t = self.log_t
        dist = self.log_dist
        vs_est = self.log_vs_est

        if self.v_cmd_const is None:
            self.get_logger().warn("v_cmd_const is None; KPI skipped.")
            return None

        t = np.asarray(t)
        dist = np.asarray(dist)
        vs_est = np.asarray(vs_est)

        mask = np.isfinite(dist) & np.isfinite(vs_est)
        if np.sum(mask) < 20:
            self.get_logger().warn("Not enough valid samples for KPI.")
            return None

        dist = dist[mask]
        vs_est = vs_est[mask]

        # --- position KPIs ---
        pos_mean = float(np.mean(dist))
        pos_rmse = float(np.sqrt(np.mean(dist**2)))
        pos_p95  = float(np.percentile(dist, 95))

        # --- speed KPIs ---
        vs_err = np.abs(vs_est - float(self.v_cmd_const))
        vs_mean = float(np.mean(vs_err))
        vs_p95  = float(np.percentile(vs_err, 95))

        kpi = {
            "v_cmd [m/s]": self.v_cmd_const,
            "N": int(len(dist)),
            "pos_mean [m]": pos_mean,
            "pos_rmse [m]": pos_rmse,
            "pos_p95 [m]": pos_p95,
            "|vs-vcmd|_mean [m/s]": vs_mean,
            "|vs-vcmd|_p95 [m/s]": vs_p95,
        }


        print("=== Perception KPI (constant commanded speed) ===")
        for k, v in kpi.items():
            print(f"{k:>22}: {v}")

        self.append_kpi_to_csv(kpi)

        return kpi
    
    def append_kpi_to_csv(self, kpi: dict):
        """
        Append one KPI record to a CSV file (no overwrite).
        If file doesn't exist, write header first.
        """
        import os
        import csv
        import datetime

        if kpi is None:
            return

        kpi_log_path = os.path.join(self.save_dir, 'kpi_log.csv')
        os.makedirs(os.path.dirname(kpi_log_path), exist_ok=True)

        file_exists = os.path.isfile(kpi_log_path)

        row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "track": self.track_name,
            "v_cmd": self.v_cmd_const,
            "N": kpi.get("N", None),
            "pos_mean": kpi.get("pos_mean [m]", None),
            "pos_rmse": kpi.get("pos_rmse [m]", None),
            "pos_p95": kpi.get("pos_p95 [m]", None),
            "vs_err_mean": kpi.get("|vs-vcmd|_mean [m/s]", None),
            "vs_err_p95": kpi.get("|vs-vcmd|_p95 [m/s]", None),
        }

        fieldnames = list(row.keys())

        with open(kpi_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Use print to avoid rosout issues after shutdown (optional)
        print(f"[KPI] Appended to {kpi_log_path}: track={self.track_name}, v_cmd={self.v_cmd_const}")


    def run_final_evaluation(self):

        self.show_main_3panel()
        self.show_hexbin_95ellipse()
        self.show_track_overlay()

        self.compute_kpi_by_speed()


def main():
    rclpy.init()
    node = EkfVsGtMonitor(track_name='Autodrive', v_cmd_const=1.0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.run_final_evaluation()

        except Exception as e:
            node.get_logger().error(f"Final evaluation failed: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
