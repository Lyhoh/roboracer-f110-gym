#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="/home/lyh/ros2_ws/src/f110_gym/localization/static_map/static_hits_raw.npz",
                        help="npz file produced by build_static_map (with dbg_*).")
    args = parser.parse_args()

    data = np.load(args.raw)

    dbg_s = data["dbg_s"]
    dbg_d = data["dbg_d"]
    dbg_x = data["dbg_x"]
    dbg_y = data["dbg_y"]
    track_length = float(data["track_length"])

    print(f"Loaded {dbg_s.size} debug points")

    # ---------- 1) s-d 平面散点图 ----------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(dbg_s, dbg_d, s=2, alpha=0.5)
    plt.xlabel("s [m]")
    plt.ylabel("d [m]")
    plt.title("Debug points in Frenet (s, d)")
    plt.grid(True)

    # 高亮 |d| > 某阈值的点（比如 5m）
    big_d_mask = np.abs(dbg_d) > 5.0
    if np.any(big_d_mask):
        plt.scatter(dbg_s[big_d_mask], dbg_d[big_d_mask],
                    s=4, alpha=0.8, c="red", label="|d| > 5")
        plt.legend()

    # ---------- 2) x-y 平面散点图 ----------
    plt.subplot(1, 2, 2)
    plt.scatter(dbg_x, dbg_y, s=2, alpha=0.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title("Debug points in map (x, y)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------- 3) s 方向统计：每个 s 段有多少点 ----------
    plt.figure(figsize=(10, 4))
    num_bins = int(track_length / 0.2)  # 每 0.2 m 一个 bin，可自己调
    hist, bin_edges = np.histogram(dbg_s, bins=num_bins, range=(0.0, track_length))

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, hist)
    plt.xlabel("s [m]")
    plt.ylabel("#points (sampled)")
    plt.title("Debug point counts along s")
    plt.grid(True)
    plt.show()

    print(f"s histogram: min={hist.min()}, max={hist.max()}, mean={hist.mean():.1f}")


if __name__ == "__main__":
    main()
