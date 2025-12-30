# Roboracer Interfaces

This package contains **custom ROS 2 message definitions** used across the Roboracer project.

The messages define shared data structures for communication between perception, planning, control, and localization modules.

Message structures and field definitions can be found directly in the corresponding `.msg` files.

---

## Message Types

- `Waypoint.msg`  
- `WaypointArray.msg`

- `Obstacle.msg`  
- `ObstacleArray.msg`

- `OTWpntArray.msg`  
  Waypoints for local overtaking trajectories.

- `Ready.msg`  
  Lightweight readiness / synchronization signal between modules.
