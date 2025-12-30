# Local Planner Package

This package implements a lightweight **local overtaking planner** based on the Frenet frame.
Its responsibility is to generate a short-horizon avoidance trajectory when a dynamic obstacle is detected ahead.

The planner is designed to be simple and robust, serving as a bridge between perception and control.

---

## Node

### `local_planner.py` — `SimpleFrenetAvoidanceNode`

**Subscribed Topics**
- `/ego_frenet` (`nav_msgs/Odometry`)  
  Ego vehicle state in Frenet coordinates (`s` in x, `d` in y).

- `/global_centerline` (`WaypointArray`)  
  Reference centerline with track boundaries (`d_left`, `d_right`), used to:
  - initialize the Frenet frame,
  - provide track limits for free-space reasoning.

- `/global_raceline` (`WaypointArray`)  
  Interface for the reference raceline used to provide baseline `d(s)` and speed information.  
  In the current setup, this topic reuses the centerline for improved stability; a true raceline can be plugged in later without changing the planner logic.

- `/perception/obstacles` (`ObstacleArray`)  
  Detected obstacles in Frenet coordinates.

**Published Topics**
- `/planner/avoidance/otwpnts` (`OTWpntArray`)  
  Local overtaking waypoints (`x`, `y`, `s`, `d`) with optional speed profile (`vx_mps`).

- `/planner/avoidance/path` (`nav_msgs/Path`)  
  Visualization of the avoidance trajectory in RViz.

- `/planner/avoidance/markers` (`MarkerArray`)  
  Point-wise visualization along the avoidance path.

- `/local_planner/ready` (`Ready`)  
  Published once the reference path is available.

---

## Planning Logic (High-Level)

1. Work entirely in the Frenet frame `(s, d)`.
2. Select the closest relevant obstacle ahead within a configurable lookahead distance.
3. Decide the overtaking side (left / right) based on available free space between obstacle extent and track boundaries.
4. Generate a smooth lateral offset profile `d(s)`:
   - start from the current lateral position,
   - move to an apex offset around the obstacle,
   - return to the reference path after passing.
5. Convert `(s, d)` back to Cartesian coordinates `(x, y)` and publish the result.
6. Assign an overtaking speed profile based on the reference speed and enforce a minimum speed advantage over the opponent.

---

## Key Parameters

**Geometry / Shape**
- `lookahead` (m)
- `evasion_dist` (m)
- `back_to_raceline_before`, `back_to_raceline_after` (m)
- `avoidance_resolution` (number of waypoints)

**Robustness / Behavior**
- `only_dynamic_obstacles`
- `ot_lost_timeout_s`
- `ot_finish_margin_s`
- `ot_cooldown_s`

**Overtake Speed Rule**
- `ot_min_speed_delta` (m/s)
- `ot_speed_scale`
- `ot_speed_cap` (m/s)

---

## Notes

- The current implementation uses **centerline as a proxy for raceline** to favor stability.
- The planner interface explicitly separates centerline and raceline inputs, enabling
  future integration of a true optimal raceline without restructuring the code.
- This module focuses on single-opponent overtaking in simulation and can be extended
  to more advanced planning methods if needed.
