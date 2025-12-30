# Global Planner Package

This package currently provides a minimal “global planner” interface by **publishing pre-defined trajectories** as ROS 2 topics.  
It does **not** perform online planning yet, but it can be extended to do so.

---

## Node

### `waypoints_from_csv.py`
Loads track waypoints from CSV files and publishes:

- **`/global_centerline`** (`roboracer_interfaces/WaypointArray`)  
  Centerline used as the Frenet reference frame and includes track boundaries (`d_left`, `d_right`).

- **`/global_raceline`** (`roboracer_interfaces/WaypointArray`)  
  Raceline projected into the **same Frenet frame** defined by the centerline (`s_m`, `d_m`).  
  If available in the CSV, it also provides `psi`, `kappa`, and speed profile (`vx`, `ax`).

Additionally, it publishes a simple RViz marker for visualization:
- **`/planner/global_waypoints_markers`** (`visualization_msgs/Marker`)

---

## Data Source

The node reads track CSVs from the `f1tenth_gym_ros` package share directory:

- `<map_name>/<map_name>_centerline.csv`
- `<map_name>/<map_name>_raceline.csv`

---

## Parameters

- `map_name` (default: `Austin`)  
  Track name (folder + file prefix).

- `frame_id` (default: `map`)  
  Frame used in published messages.

- `publish_rate` (default: `0.5` Hz)  
  Publish frequency. Set to `0` to publish once.

---

## Notes

- This package is a trajectory publisher for now; actual online global planning can be added later if needed.
