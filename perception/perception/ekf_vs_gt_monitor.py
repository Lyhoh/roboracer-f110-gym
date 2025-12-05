"""
EKF vs GT online monitor (using your FrenetConverter).
Subscribes:
  - /global_centerline (for converter)
  - /opp_racecar/odom  (GT in world frame)
  - /perception/obstacles (EKF estimate in Frenet)
Publishes:
  - /monitor/err_cart : Float32MultiArray [ex, ey, dist]
  - /monitor/err_fren : Float32MultiArray [e_s, e_vs, e_d, e_vd]
"""

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
from interfaces.msg import ObstacleArray, WaypointArray
from std_msgs.msg import Float32MultiArray

from perception.frenet_converter import FrenetConverter 


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
    # Differentiate with wrap-around (assume racetrack is closed)
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)

    psi = np.arctan2(dy, dx)  # atan2 gives angle in [-pi, pi]
    
    # Optional smoothing if your path is noisy
    # psi = np.unwrap(psi)  # continuous heading
    # psi = np.convolve(psi, np.ones(5)/5, mode='same')

    return psi


class EkfVsGtMonitor(Node):
    def __init__(self):
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


        self.get_logger().info('[EkfVsGtMonitor] node started.')

    # ---------- Waypoints → FrenetConverter ----------
    def cb_wp(self, msg: WaypointArray):
        if not msg.wpnts:
            return

        x = np.array([w.x_m for w in msg.wpnts], dtype=float)
        y = np.array([w.y_m for w in msg.wpnts], dtype=float)

        # Optional: psi per waypoint, if present in your message; else keep None
        # try:
        #     psi = np.array([w.psi_rad for w in msg.wpnts], dtype=float)  # adjust field name if you have it
        # except Exception:
        #     psi = None
        psi = compute_waypoints_psi(x, y)

        self.fc = FrenetConverter(waypoints_x=x, waypoints_y=y, waypoints_psi=psi)
        self.track_length = float(self.fc.raceline_length)
        self.get_logger().info(f'[Monitor] FrenetConverter ready. L={self.track_length:.2f} m.')

        # --- directly cache the centerline from waypoints ---
        try:
            self.centerline_xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=float)
            self.get_logger().info(f"[Monitor] Cached {len(self.centerline_xy)} centerline points from waypoints.")
        except Exception as e:
            self.centerline_xy = None
            self.get_logger().warn(f"[Monitor] Failed to cache centerline: {e}")


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
        self.pub_err_cart.publish(msg_cart)

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
        self.pub_err_fren.publish(msg_fren)

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
            # self.get_logger().info(
            #     f"[N={self.n}] Frenet "
            #     f"e_s: bias={bias[0]:+.3f} mae={mae[0]:.3f} rmse={rmse[0]:.3f} | "
            #     f"e_vs: bias={bias[1]:+.3f} mae={mae[1]:.3f} rmse={rmse[1]:.3f} | "
            #     f"e_d: bias={bias[2]:+.3f} mae={mae[2]:.3f} rmse={rmse[2]:.3f} | "
            #     f"e_vd: bias={bias[3]:+.3f} mae={mae[3]:.3f} rmse={rmse[3]:.3f} | "
            #     f"Cart dist mean≈{float(np.mean(np.sqrt(self.sum_e2[2]/self.n))):.3f}"
            # )

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


    def show_figure(self):
        """Show a single figure with multiple subplots (no saving)."""
        if len(self.log_t) == 0:
            self.get_logger().warn("No samples collected; nothing to display.")
            return

        t   = np.array(self.log_t)
        es  = np.array(self.log_es)
        evs = np.array(self.log_evs)
        ed  = np.array(self.log_ed)
        evd = np.array(self.log_evd)
        dist= np.array(self.log_dist)
        xy_ref = np.array(self.log_xy_ref)  # shape (N, 2)
        xy_est = np.array(self.log_xy_est)  # shape (N, 2)

        # Optional: downsample for lighter plotting if very long
        step = max(1, len(t) // 4000)
        t, es, evs, ed, evd, dist = t[::step], es[::step], evs[::step], ed[::step], evd[::step], dist[::step]
        xy_ref = xy_ref[::step]
        xy_est = xy_est[::step]

        # Build a 2x3 grid: (e_s, e_vs, e_d) / (e_vd, cart_dist, XY track)
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)
        # fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        # gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])  

        ax_es  = fig.add_subplot(gs[0, 0])
        ax_evs = fig.add_subplot(gs[0, 1])
        ax_ed  = fig.add_subplot(gs[0, 2])
        ax_evd = fig.add_subplot(gs[1, 0])
        ax_dst = fig.add_subplot(gs[1, 1])
        # ax_xy  = fig.add_subplot(gs[1, 2])

        # e_s (wrapped)
        ax_es.plot(t, es)
        ax_es.set_title("e_s (wrapped)")
        ax_es.set_xlabel("time [s]")
        ax_es.set_ylabel("m")

        # e_vs
        ax_evs.plot(t, evs)
        ax_evs.set_title("e_vs")
        ax_evs.set_xlabel("time [s]")
        ax_evs.set_ylabel("m/s")

        # e_d
        ax_ed.plot(t, ed)
        ax_ed.set_title("e_d")
        ax_ed.set_xlabel("time [s]")
        ax_ed.set_ylabel("m")

        # e_vd
        ax_evd.plot(t, evd)
        ax_evd.set_title("e_vd")
        ax_evd.set_xlabel("time [s]")
        ax_evd.set_ylabel("m/s")

        # Cartesian distance
        ax_dst.plot(t, dist)
        ax_dst.set_title("Cartesian distance error")
        ax_dst.set_xlabel("time [s]")
        ax_dst.set_ylabel("m")

        # # XY overlay (reference vs estimate)
        # ax_xy.plot(xy_ref[:,0], xy_ref[:,1], lw=1, label="GT XY")
        # ax_xy.plot(xy_est[:,0], xy_est[:,1], lw=1, label="EKF XY")
        # ax_xy.set_title("Track (XY) overlay")
        # ax_xy.set_xlabel("x [m]")
        # ax_xy.set_ylabel("y [m]")
        # ax_xy.axis("equal")
        # ax_xy.legend(loc="best")

        # --- (ex, ey) scatter ---
        ex = np.array(self.log_ex)[::step]
        ey = np.array(self.log_ey)[::step]
        mask = np.isfinite(ex) & np.isfinite(ey)
        ex, ey = ex[mask], ey[mask]

        ax_xy = fig.add_subplot(gs[1, 2])
        # ax_xy = fig.add_subplot(gs[1, 1:3]) 

        hb = ax_xy.hexbin(ex, ey, gridsize=40, cmap='Blues', bins='log', mincnt=1)
        cb = fig.colorbar(hb, ax=ax_xy)
        cb.set_label('count (log scale)')

        mx, my = ex.mean(), ey.mean()
        cov = np.cov(np.vstack([ex, ey]))
        vals, vecs = np.linalg.eigh(cov)
        t = np.linspace(0, 2*np.pi, 256)
        ell = vecs @ (np.sqrt(vals)[:,None] * np.vstack([np.cos(t), np.sin(t)]))
        ax_xy.plot(ell[0] + mx, ell[1] + my, 'r--', lw=2.0, label="1σ ellipse")
        ax_xy.scatter([mx], [my], marker='x', s=90, color='r', label='mean')

        xlo, xhi = np.percentile(ex, [1, 99])
        ylo, yhi = np.percentile(ey, [1, 99])
        padx = 0.5 * max(abs(xlo), abs(xhi))
        pady = 0.15 * max(abs(ylo), abs(yhi))
        ax_xy.set_xlim(xlo - padx, xhi + padx)
        ax_xy.set_ylim(ylo - pady, yhi + pady)
        # xlo, xhi = np.percentile(ex, [1, 99]); ylo, yhi = np.percentile(ey, [1, 99])
        # padx = 0.15*max(abs(xlo), abs(xhi)); pady = 0.15*max(abs(ylo), abs(yhi))
        # ax_xy.set_xlim(xlo-padx, xhi+padx); ax_xy.set_ylim(ylo-pady, yhi+pady)
        # ax_xy.set_aspect('equal', 'box')
        ax_xy.grid(True, alpha=0.25)
        ax_xy.set_title("Cartesian Error Density (hexbin)")
        ax_xy.set_xlabel("Error X [m]"); ax_xy.set_ylabel("Error Y [m]")
        ax_xy.legend(loc="best")


        # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # fname = f"ekf_eval_{ts}.png"
        fname = "ekf_vs_gt_monitor_error.png"
        path = os.path.join('/home/lyh/ros2_ws/src/f110_gym/perception/results/', fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved summary figure: {path}")

        # plt.show()
        plt.show(block=False)
        plt.pause(0.1)  

    def show_compare_figure(self):
        """Show second big figure: GT vs EKF (s, d, vs, vd) + X/Y coordinate comparison + track overlay."""
        if len(self.log_t) == 0 or len(self.log_s_true) == 0:
            self.get_logger().warn("No samples collected for comparison; nothing to display.")
            return

        t = np.array(self.log_t)
        s_true = np.array(self.log_s_true); s_est = np.array(self.log_s_est)
        d_true = np.array(self.log_d_true); d_est = np.array(self.log_d_est)
        vs_true = np.array(self.log_vs_true); vs_est = np.array(self.log_vs_est)
        vd_true = np.array(self.log_vd_true); vd_est = np.array(self.log_vd_est)

        # XY coordinates and dist
        xy_ref = np.array(self.log_xy_ref)  # GT
        xy_est = np.array(self.log_xy_est)  # EKF

        # Downsample if needed
        step = max(1, len(t)//4000)
        t = t[::step]
        s_true, s_est = s_true[::step], s_est[::step]
        d_true, d_est = d_true[::step], d_est[::step]
        vs_true, vs_est = vs_true[::step], vs_est[::step]
        vd_true, vd_est = vd_true[::step], vd_est[::step]
        xy_ref = xy_ref[::step]; xy_est = xy_est[::step]

        # --- Figure layout: 3x2 ---
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2)

        ax_s   = fig.add_subplot(gs[0, 0])
        ax_d   = fig.add_subplot(gs[0, 1])
        ax_vs  = fig.add_subplot(gs[1, 0])
        ax_vd  = fig.add_subplot(gs[1, 1])
        ax_xyc = fig.add_subplot(gs[2, 0])  # now X/Y coordinate comparison
        ax_xy  = fig.add_subplot(gs[2, 1])  # track overlay (unchanged)

        # --- s comparison ---
        ax_s.plot(t, s_true, label="GT s")
        ax_s.plot(t, s_est,  label="EKF s", alpha=0.9)
        ax_s.set(title="s comparison", xlabel="time [s]", ylabel="s [m]")
        ax_s.legend(loc="best")

        # --- d comparison ---
        ax_d.plot(t, d_true, label="GT d")
        ax_d.plot(t, d_est,  label="EKF d", alpha=0.9)
        ax_d.set(title="d comparison", xlabel="time [s]", ylabel="d [m]")
        ax_d.legend(loc="best")

        # --- vs comparison ---
        ax_vs.plot(t, vs_true, label="GT vs")
        ax_vs.plot(t, vs_est,  label="EKF vs", alpha=0.9)
        ax_vs.set(title="vs comparison", xlabel="time [s]", ylabel="m/s")
        ax_vs.legend(loc="best")

        # --- vd comparison ---
        ax_vd.plot(t, vd_true, label="GT vd")
        ax_vd.plot(t, vd_est,  label="EKF vd", alpha=0.9)
        ax_vd.set(title="vd comparison", xlabel="time [s]", ylabel="m/s")
        ax_vd.legend(loc="best")

        # --- X/Y coordinate comparison (time series) ---
        x_gt, y_gt = xy_ref[:, 0], xy_ref[:, 1]
        x_est, y_est = xy_est[:, 0], xy_est[:, 1]

        ax_xyc.plot(t, x_gt, color="tab:blue", linestyle="-",  label="GT x")
        ax_xyc.plot(t, x_est, color="tab:blue", linestyle="--", label="EKF x")
        ax_xyc.plot(t, y_gt, color="tab:orange", linestyle="-",  label="GT y")
        ax_xyc.plot(t, y_est, color="tab:orange", linestyle="--", label="EKF y")
        ax_xyc.set(title="Cartesian coordinates comparison (x,y)",
                xlabel="time [s]", ylabel="[m]")
        ax_xyc.legend(loc="best")

        # --- Track overlay (GT vs EKF) ---
        ax_xy.plot(xy_ref[:, 0], xy_ref[:, 1], lw=1, label="GT XY")
        ax_xy.plot(xy_est[:, 0], xy_est[:, 1], lw=1, label="EKF XY")
        ax_xy.set(title="Track overlay (GT vs EKF)", xlabel="x [m]", ylabel="y [m]")
        ax_xy.axis("equal")
        ax_xy.legend(loc="best")

        # Save
        save_dir = '/home/lyh/ros2_ws/src/f110_gym/perception/results/'
        os.makedirs(save_dir, exist_ok=True)
        fname = "ekf_vs_gt_monitor_compare.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved comparison figure: {os.path.join(save_dir, fname)}")
        
        # plt.show()
        plt.show(block=False)
        plt.pause(0.1)

    def show_det_vs_trk_figure(self):
        """Show/save a figure comparing Detection vs Tracking in XY and Frenet."""
        if len(self.log_det_dist) == 0:
            self.get_logger().warn("No detection samples to compare; nothing to display.")
            return

        t = np.array(self.log_t)
        # Make sure all arrays have same length (guard for occasional missing det frames)
        n = min(len(t),
                len(self.log_det_dist),
                len(self.log_det_es), len(self.log_det_ed),
                len(self.log_xy_est), len(self.log_xy_det))
        t = t[:n]
        det_dist = np.array(self.log_det_dist[:n])
        e_s_dt   = np.array(self.log_det_es[:n])
        e_d_dt   = np.array(self.log_det_ed[:n])

        xy_trk = np.array(self.log_xy_est[:n])
        xy_det = np.array(self.log_xy_det[:n])

        # light downsample if long
        step = max(1, len(t)//4000)
        t, det_dist, e_s_dt, e_d_dt = t[::step], det_dist[::step], e_s_dt[::step], e_d_dt[::step]
        xy_trk, xy_det = xy_trk[::step], xy_det[::step]

        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)

        # Left: time series (distance + Frenet residuals)
        ax_ts = fig.add_subplot(gs[0, 0])
        ax_ts.plot(t, det_dist, label="|det - trk| in XY [m]")
        ax_ts.plot(t, e_s_dt,  label="e_s (det - trk) [m]")
        ax_ts.plot(t, e_d_dt,  label="e_d (det - trk) [m]")
        ax_ts.set_title("Detection vs Tracking (time series)")
        ax_ts.set_xlabel("time [s]")
        ax_ts.legend(loc="best")
        ax_ts.grid(True, alpha=0.3)

        # Right: XY overlay of detection vs tracking (+GT)
        ax_xy = fig.add_subplot(gs[0, 1])

        # Tracking (EKF)
        ax_xy.plot(xy_trk[:, 0], xy_trk[:, 1], lw=1.2, color='tab:blue', label="Tracking (EKF)")

        # Detection (raw)
        ax_xy.plot(xy_det[:, 0], xy_det[:, 1], lw=1.0, color='tab:orange', label="Detection (raw)")

        # Ground truth (GT) — use the same downsample as others
        xy_gt = np.array(self.log_xy_ref[:n])[::step]
        ax_xy.plot(xy_gt[:, 0], xy_gt[:, 1], lw=1.2, color='tab:green', label="Ground Truth")

        # Plot formatting
        ax_xy.set_title("XY overlay: Detection vs Tracking vs GT")
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.axis("equal")
        ax_xy.grid(True, alpha=0.25)
        ax_xy.legend(loc="best")

        # # Right: XY overlay of detection vs tracking (+GT + centerline)
        # ax_xy = fig.add_subplot(gs[0, 1])

        # # --- draw centerline if available ---
        # if self.centerline_xy is not None and len(self.centerline_xy) > 1:
        #     ax_xy.plot(self.centerline_xy[:, 0], self.centerline_xy[:, 1],
        #             lw=1.0, color='0.6', alpha=0.8, zorder=0, label="Centerline")

        # # Tracking (EKF)
        # ax_xy.plot(xy_trk[:, 0], xy_trk[:, 1], lw=1.2, color='tab:blue', zorder=2, label="Tracking (EKF)")
        # # Detection (raw)
        # ax_xy.plot(xy_det[:, 0], xy_det[:, 1], lw=1.0, color='tab:orange', zorder=3, label="Detection (raw)")
        # # Ground truth (GT)
        # xy_gt = np.array(self.log_xy_ref[:n])[::step]
        # ax_xy.plot(xy_gt[:, 0], xy_gt[:, 1], lw=1.2, color='tab:green', zorder=4, label="Ground Truth")

        # ax_xy.set_title("XY overlay: Detection vs Tracking vs GT (+Centerline)")
        # ax_xy.set_xlabel("x [m]")
        # ax_xy.set_ylabel("y [m]")
        # ax_xy.axis("equal")
        # ax_xy.grid(True, alpha=0.25)
        # ax_xy.legend(loc="best")


        # Save next to your other outputs
        save_dir = '/home/lyh/ros2_ws/src/f110_gym/perception/results/'
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "det_vs_trk.png"), dpi=150, bbox_inches="tight")
        self.get_logger().info(f"Saved detection-vs-tracking figure to {os.path.join(save_dir, 'det_vs_trk.png')}")
        plt.show()




def main():
    rclpy.init()
    node = EkfVsGtMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pass
            # node.show_figure()  # <-- show on Ctrl+C
            # node.show_compare_figure()
            # node.show_det_vs_trk_figure()

        except Exception as e:
            node.get_logger().error(f"Failed to show figure: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
