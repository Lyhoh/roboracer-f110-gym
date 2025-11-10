"""
Offline evaluation (no CLI): 
- Reference from waypoints CSV: x_m,y_m,s_m,d_left,d_right
- Constant tangential speed VS_CONST
- Reference d = D0 (centerline by default)
- Align to EKF CSV timestamps and compute errors:
  * Frenet errors: e_s (wrapped), e_d, e_vs, e_vd
  * Cartesian position error (Euclidean) by projecting (s,d) to (x,y) via path normal
"""

from __future__ import annotations
import csv
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIG ============================
EKF_CSV = '/home/lyh/ros2_ws/src/f110_gym/perception/results/obstacles_ekf.csv'   # Path to EKF log CSV: columns t,s,d,vs,vd,...
WP_CSV  = '/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_waypoints.csv'       # Path to waypoints CSV: x_m,y_m,s_m,d_left,d_right

VS_CONST = 1.0                  # Constant reference tangential speed [m/s]
D0       = 0.0                  # Reference lateral offset [m], 0 = centerline

S0_MODE  = "manual"          # "ekf_first" | "zero" | "manual"
S0_VALUE = 0.0 # 24.73931542          # Used only when S0_MODE == "manual"

SHOW_PLOTS = True               # If True, show matplotlib plots
# =============================================================

# ---------- ring helpers ----------
def wrap_s_residual(ds: np.ndarray | float, L: float):
    """Shortest signed residual on ring of length L."""
    if L is None or L <= 0.0: return ds
    return (ds + 0.5*L) % L - 0.5*L

def normalize_s(sv: np.ndarray | float, L: float):
    """Map s into [0, L)."""
    if L is None or L <= 0.0: return sv
    return sv % L

# ---------- data I/O ----------
def load_ekf_csv(path: str):
    """Load EKF CSV with columns: t,s,d,vs,vd,(...ignored)."""
    t, s, d, vs, vd = [], [], [], [], []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row['t']))
            s.append(float(row['s']))
            d.append(float(row['d']))
            vs.append(float(row['vs']))
            vd.append(float(row['vd']))
    t = np.array(t); s = np.array(s); d = np.array(d)
    vs = np.array(vs); vd = np.array(vd)
    order = np.argsort(t)
    return t[order], s[order], d[order], vs[order], vd[order]

def load_waypoints_csv(path: str):
    """Load waypoints CSV with columns: x_m,y_m,s_m,d_left,d_right."""
    xs, ys, ss, dl, dr = [], [], [], [], []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row['x_m']))
            ys.append(float(row['y_m']))
            ss.append(float(row['s_m']))
            dl.append(float(row['d_left']))
            dr.append(float(row['d_right']))
    x = np.array(xs); y = np.array(ys); s = np.array(ss)
    d_left = np.array(dl); d_right = np.array(dr)
    order = np.argsort(s)
    return s[order], x[order], y[order], d_left[order], d_right[order]

# ---------- periodic path interpolation ----------
def build_periodic_path(s_wp: np.ndarray, x_wp: np.ndarray, y_wp: np.ndarray):
    """
    Build periodic interpolators for centerline position and tangent.
    Piecewise-linear interpolation over s, extended periodically.
    """
    L = float(s_wp[-1])  # assume s starts near 0 and ends at track length
    dx_ds = np.gradient(x_wp, s_wp)
    dy_ds = np.gradient(y_wp, s_wp)

    # Periodic extension: [s-L, s, s+L]
    s_ext = np.concatenate([s_wp - L, s_wp, s_wp + L])
    x_ext = np.concatenate([x_wp,     x_wp, x_wp    ])
    y_ext = np.concatenate([y_wp,     y_wp, y_wp    ])
    dxd_ext = np.concatenate([dx_ds,  dx_ds, dx_ds  ])
    dyd_ext = np.concatenate([dy_ds,  dy_ds, dy_ds  ])

    def interp_centerline(sq: np.ndarray):
        # Map queries to [0,L) then shift to middle block
        sqn = normalize_s(sq, L) + L
        xq = np.interp(sqn, s_ext, x_ext)
        yq = np.interp(sqn, s_ext, y_ext)
        tx = np.interp(sqn, s_ext, dxd_ext)
        ty = np.interp(sqn, s_ext, dyd_ext)
        # Normalize tangent and build left-normal [-ty, tx]
        tnorm = np.hypot(tx, ty) + 1e-12
        tx /= tnorm; ty /= tnorm
        nx, ny = -ty, tx
        return xq, yq, tx, ty, nx, ny

    return L, interp_centerline

# ---------- metrics ----------
def summarize(e: np.ndarray):
    """Return bias, MAE, RMSE, std (zero-mean), and max_abs."""
    bias = float(np.mean(e))
    mae  = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    std  = float(np.std(e - bias))
    mxe  = float(np.max(np.abs(e)))
    return dict(bias=bias, mae=mae, rmse=rmse, std=std, max_abs=mxe)

# ---------- main ----------
def main():
    # Load data
    t, s_est, d_est, vs_est, vd_est = load_ekf_csv(EKF_CSV)
    s_wp, x_wp, y_wp, dL_wp, dR_wp  = load_waypoints_csv(WP_CSV)

    # Build periodic centerline interpolator
    L, interp = build_periodic_path(s_wp, x_wp, y_wp)

    # Reference s(t)
    t0 = t[0]
    if S0_MODE == "ekf_first":
        s0 = float(normalize_s(s_est[0], L))
    elif S0_MODE == "zero":
        s0 = 0.0
    elif S0_MODE == "manual":
        s0 = float(S0_VALUE)
    else:
        raise ValueError("S0_MODE must be 'ekf_first', 'zero', or 'manual'.")

    s_true  = normalize_s(s0 + VS_CONST * (t - t0), L)
    d_true  = np.full_like(t, D0)
    vs_true = np.full_like(t, VS_CONST)
    vd_true = np.zeros_like(t)

    # Frenet errors
    e_s  = wrap_s_residual(s_est - s_true, L)
    e_d  = d_est - d_true
    e_vs = vs_est - vs_true
    e_vd = vd_est - vd_true

    print("== Frenet metrics ==")
    print("s  :", summarize(e_s))
    print("d  :", summarize(e_d))
    print("vs :", summarize(e_vs))
    print("vd :", summarize(e_vd))

    # Cartesian position error: project (s,d) to (x,y) using centerline + normal
    x_ct, y_ct, _, _, nx_t, ny_t = interp(s_true)                 # centerline at s_true
    x_ce, y_ce, _, _, nx_e, ny_e = interp(normalize_s(s_est, L))  # centerline at s_est

    x_true = x_ct + nx_t * d_true
    y_true = y_ct + ny_t * d_true
    x_hat  = x_ce + nx_e * d_est
    y_hat  = y_ce + ny_e * d_est

    pos_err = np.hypot(x_hat - x_true, y_hat - y_true)
    print("== Cartesian position error (m) ==")
    print(summarize(pos_err))

    if SHOW_PLOTS:
        # s vs time
        plt.figure(figsize=(10,4))
        plt.plot(t, s_true, label="s_true")
        plt.plot(t, s_est,  label="s_est", alpha=0.85)
        plt.title("s"); plt.xlabel("t [s]"); plt.ylabel("s [m]"); plt.legend(); plt.tight_layout()

        # e_s
        plt.figure(figsize=(10,3))
        plt.plot(t, e_s); plt.title("e_s (wrapped)"); plt.xlabel("t [s]"); plt.ylabel("m"); plt.tight_layout()

        # # d vs time
        # plt.figure(figsize=(10,4))
        # plt.plot(t, d_true, label="d_true")
        # plt.plot(t, d_est,  label="d_est", alpha=0.85)
        # plt.title("d"); plt.xlabel("t [s]"); plt.ylabel("m"); plt.legend(); plt.tight_layout()

        # # e_d
        # plt.figure(figsize=(10,3))
        # plt.plot(t, e_d); plt.title("e_d"); plt.xlabel("t [s]"); plt.ylabel("m"); plt.tight_layout()

        # # vs
        # plt.figure(figsize=(10,4))
        # plt.plot(t, vs_true, label="vs_true")
        # plt.plot(t, vs_est,  label="vs_est", alpha=0.85)
        # plt.title("vs"); plt.xlabel("t [s]"); plt.ylabel("m/s"); plt.legend(); plt.tight_layout()

        # # e_vs
        # plt.figure(figsize=(10,3))
        # plt.plot(t, e_vs); plt.title("e_vs"); plt.xlabel("t [s]"); plt.ylabel("m/s"); plt.tight_layout()

        # # 2D track view + samples
        # step = max(1, len(t)//200)
        # plt.figure(figsize=(6,6))
        # plt.plot(x_wp, y_wp, '-', lw=1, label='centerline')
        # plt.scatter(x_true[::step], y_true[::step], s=8, label='ref (samples)')
        # plt.scatter(x_hat[::step],  y_hat[::step],  s=8, label='EKF (samples)', alpha=0.8)
        # plt.axis('equal'); plt.legend(); plt.title("Track (XY) & samples"); plt.tight_layout()

        # # position error over time
        # plt.figure(figsize=(10,3))
        # plt.plot(t, pos_err)
        # plt.title("Cartesian position error"); plt.xlabel("t [s]"); plt.ylabel("m"); plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()
