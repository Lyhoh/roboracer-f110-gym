# Localization Package

This package provides **offline utilities for building a static track wall map** from LiDAR data.
The generated static map can be used by the perception module for **optional track-based filtering**.

This package does **not** perform online localization.

---

## Purpose

The goal of this package is to extract a **static representation of track boundaries** (left and right walls in Frenet coordinates) by accumulating LiDAR observations over time.

The resulting static map serves as a lightweight prior that can help:
- suppress false obstacle detections near track boundaries,
- distinguish static track structure from dynamic objects.

---

## Components

### `build_static_map.py`

A ROS 2 node that builds a raw static wall dataset by accumulating LiDAR hits.

**Functionality**
- Subscribes to:
  - `/global_centerline` (reference path for Frenet frame),
  - `/ego_racecar/odom`,
  - `/scan`.
- Transforms LiDAR points into the map frame.
- Projects points into Frenet `(s, d)` coordinates.
- Accumulates near-track hits over multiple frames.

**Output (on shutdown)**
- `static_hits_raw.npz`  
  Raw accumulated `(s, d)` wall points and metadata.

---

### `fit_static_walls.py`

An offline post-processing script to convert raw LiDAR hits into a clean static wall map.

**Functionality**
- Loads `static_hits_raw.npz`.
- Bins points along the `s` axis.
- Estimates left/right walls using median statistics.
- Applies smoothing and gap filling.
- Outputs a continuous wall representation.

**Output**
- `static_map.npz`  
  Fitted static walls:
  - `s_axis`
  - `d_left(s)`
  - `d_right(s)`

---

## Notes

- This package is intended for **offline, pre-run processing**.
- The static map is generated once and reused across runs.
- Execution and launch instructions are documented in the
  `roboracer_f110_bringup` package.
- Using the static map in perception is **optional** and can be disabled without affecting system operation.

