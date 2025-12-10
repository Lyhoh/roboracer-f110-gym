#!/usr/bin/env python3
"""
Offline visualization for static map:
- Plot hit heatmap in (s, d)
- Plot extracted d_left(s) and d_right(s)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", type=str, default="/home/lyh/ros2_ws/src/f110_gym/localization/static_map/static_hits_raw.npz",
                        help="Path to raw static hits npz.")
    parser.add_argument("--map", type=str, default="/home/lyh/ros2_ws/src/f110_gym/localization/static_map/static_map.npz",
                        help="Path to final static wall map npz.")
    args = parser.parse_args()

    # Load raw hits
    hits_data = np.load(args.hits)
    static_hits = hits_data["static_hits"]   # (Ns, Nd)
    s_axis = hits_data["s_axis"]             # (Ns,)
    d_bins = hits_data["d_bins"]             # (Nd,)

    # Load final walls
    map_data = np.load(args.map)
    s_axis_map = map_data["s_axis"]
    d_left = map_data["d_left"]
    d_right = map_data["d_right"]

    # 1) Heatmap in (s, d)
    S, D = np.meshgrid(s_axis, d_bins, indexing="ij")  # S, D shape: (Ns, Nd)
    hits_log = np.log1p(static_hits.astype(np.float64))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    im = plt.pcolormesh(S, D, hits_log, shading="auto")
    plt.colorbar(im, label="log(1 + hits)")
    plt.xlabel("s [m]")
    plt.ylabel("d [m]")
    plt.title("Static hits heatmap in Frenet (s, d)")

    # 2) Wall curves d_left(s), d_right(s)
    plt.subplot(2, 1, 2)
    plt.plot(s_axis_map, d_left, label="d_left (static wall)")
    plt.plot(s_axis_map, d_right, label="d_right (static wall)")
    plt.xlabel("s [m]")
    plt.ylabel("d [m]")
    plt.legend()
    plt.grid(True)
    plt.title("Extracted static walls")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
