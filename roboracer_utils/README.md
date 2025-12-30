# Roboracer Utils

This package contains **utility scripts and helper nodes** shared across the Roboracer project.

The components in this package provide common functionality and are not standalone modules.

---

## Contents

- `frenet_converter.py`  
  Utility for converting between Cartesian and Frenet coordinates.

- `frenet_odom_republisher.py`  
  Republishes ego vehicle odometry in Frenet coordinates.

- `tf_bridge_map_odom.py`  
  Provides a TF bridge between `map` and `odom` frames for compatibility across modules.
