import os
import numpy as np
import matplotlib.pyplot as plt

def load_centerline(centerline_path: str):
    """
    Load centerline CSV.
    Format: x_m, y_m, w_tr_right_m, w_tr_left_m
    """
    data = np.genfromtxt(centerline_path,
                         delimiter=',',
                         comments='#')  # ignore lines starting with '#'
    # Columns: x, y, w_tr_right, w_tr_left
    x = data[:, 0]
    y = data[:, 1]
    w_right = data[:, 2]
    w_left = data[:, 3]
    return x, y, w_right, w_left

def load_raceline(raceline_path: str):
    """
    Load raceline CSV.
    Format: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    """
    data = np.genfromtxt(raceline_path,
                         delimiter=';',
                         comments='#')  # ignore hash-commented lines
    # Columns: s, x, y, psi, kappa, vx, ax
    x = data[:, 1]
    y = data[:, 2]
    return x, y

def compute_track_boundaries(x_c, y_c, w_right, w_left):
    """
    Compute left/right track boundaries from centerline + widths.
    We approximate tangent using finite differences and use its normal.
    """
    # Tangent vector via numerical gradient
    dx = np.gradient(x_c)
    dy = np.gradient(y_c)

    # Avoid division by zero
    eps = 1e-6
    norm = np.hypot(dx, dy) + eps

    # Unit tangent
    tx = dx / norm
    ty = dy / norm

    # Left normal (rotate tangent by +90 deg)
    nx = -ty
    ny = tx

    # Right normal is just the opposite direction
    # We will use +nx for left boundary and -nx for right boundary
    x_left = x_c + nx * w_left
    y_left = y_c + ny * w_left

    x_right = x_c - nx * w_right
    y_right = y_c - ny * w_right

    return x_left, y_left, x_right, y_right

def plot_track(centerline_path: str, raceline_path: str, title: str = ""):
    # Load data
    x_c, y_c, w_r, w_l = load_centerline(centerline_path)
    x_r, y_r = load_raceline(raceline_path)

    # Compute boundaries
    x_left, y_left, x_right, y_right = compute_track_boundaries(x_c, y_c, w_r, w_l)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_c, y_c, label="Centerline", linewidth=1.5)
    plt.plot(x_r, y_r, label="Raceline", linewidth=1.5)
    plt.plot(x_left, y_left, '--', label="Left boundary", linewidth=1)
    plt.plot(x_right, y_right, '--', label="Right boundary", linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(centerline_path), f"{title}_track.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Change these paths to match your project structure
    base_dir = "/home/lyh/ros2_ws/src/f110_gym/f1tenth_gym_ros/maps/f1tenth_racetracks"
    track_name = "IMS"

    centerline_csv = os.path.join(base_dir, track_name, f"{track_name}_centerline.csv")
    raceline_csv = os.path.join(base_dir, track_name, f"{track_name}_raceline.csv")

    plot_track(centerline_csv, raceline_csv, title=track_name)
