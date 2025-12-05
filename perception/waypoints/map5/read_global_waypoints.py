#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read centerline CSV (x_m,y_m,w_tr_right_m,w_tr_left_m),
compute s_m (cumulative arc length), and export global_waypoints.csv
with columns: x_m,y_m,s_m,d_left,d_right.

- If your header uses "#x_m", it is handled automatically.
- By default we assume an open path (not closing the loop).
  Set CLOSE_LOOP = True if your track is closed and you want s_m to wrap to total length at end.
"""

import csv
import os
import numpy as np

# ====== CONFIG ======
CENTERLINE_CSV = "/home/lyh/ros2_ws/src/f110_gym/global-planning/outputs/map5/centerline.csv"   # <-- change to your path
OUT_CSV        = "/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline.csv"
CLOSE_LOOP     = False  # True if the track is a closed loop

def load_centerline(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        rows = []
        for row in reader:
            if not row or row[0].startswith("#") and len(row) == 1:
                continue
            rows.append([float(x) for x in row])
    data = np.array(rows, dtype=np.float64)
    # Column names may be "#x_m" or "x_m"
    col = {name: i for i, name in enumerate(header)}
    x_col = col["#x_m"] if "#x_m" in col else col["x_m"]
    y_col = col["y_m"]
    wr_col = col["w_tr_right_m"]
    wl_col = col["w_tr_left_m"]
    xy = data[:, [x_col, y_col]]
    w_tr_right = data[:, wr_col]  # positive width to the right
    w_tr_left  = data[:, wl_col]  # positive width to the left
    return xy, w_tr_left, w_tr_right

def compute_s(xy, close_loop=False):
    """Compute cumulative arc length along the polyline."""
    diffs = np.diff(xy, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if close_loop:
        # Optionally add last segment back to start to compute total length,
        # but keep s as an open sequence (last s is total length).
        last_seg = np.hypot(*(xy[0] - xy[-1]))
        s[-1] = s[-2] + last_seg
    return s

def save_global_waypoints(path, xy, s, d_left, d_right):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "y_m", "s_m", "d_left", "d_right"])
        for (x, y), si, dl, dr in zip(xy, s, d_left, d_right):
            w.writerow([f"{x:.6f}", f"{y:.6f}", f"{si:.6f}", f"{dl:.6f}", f"{dr:.6f}"])
    print("Saved:", path)

def main():
    xy, w_tr_left, w_tr_right = load_centerline(CENTERLINE_CSV)

    # In Frenet-style naming: d_left/d_right are just the widths relative to centerline.
    d_left  = w_tr_left
    d_right = w_tr_right

    s_m = compute_s(xy, close_loop=CLOSE_LOOP)
    save_global_waypoints(OUT_CSV, xy, s_m, d_left, d_right)

if __name__ == "__main__":
    main()