# Roboracer Bringup

This package contains **launch files and global configuration** used to start the Roboracer system in the f1tenth_gym simulation environment.

It serves as the main entry point for running the system in different modes(e.g. full pipeline execution or offline preprocessing).

---

## Launch Files

### `full_system.launch.py`

Launches the main simulation pipeline, including:

- f1tenth_gym ROS bridge
- global planner (path publishing)
- perception module
- local planner (optional, for overtaking)
- controller

This launch file is typically used for **runtime experiments and demos**.

**Common arguments**
- `map_name`  
  Racetrack name (must exist in `f1tenth_gym_ros/maps/f1tenth_racetracks`).

- `rviz_profile`  
  RViz preset configuration (e.g. `follow`, `global`).

---

### `build_static_map.launch.py`

Runs **offline static map generation** before executing the full system.

This launch file starts:
- the simulator,
- global planner (for Frenet reference),
- localization utilities for accumulating LiDAR hits.

It is intended to be executed **once per track**, prior to runtime experiments.

---

## Maps and Configuration

- Racetrack CSV files are loaded from the `f1tenth_gym_ros` package under: `maps/f1tenth_racetracks/<map_name>/`.

- The ego vehicle start pose is configured in the `f1tenth_gym_ros` package (e.g. via `config/sim.yaml`).

---

## Notes

- This package only defines **how the system is launched**.
- Algorithmic details are documented in the corresponding packages
(perception, planning, control, localization).

