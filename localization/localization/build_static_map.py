"""
ROS2 node to build a static track wall map from LiDAR:
- Subscribes to /global_centerline, /ego_racecar/odom, /scan
- Accumulates LiDAR hits in Frenet (s, d) grid
- On shutdown (Ctrl+C), computes static left/right walls d_left(s), d_right(s)
  and saves:
    - static_hits_raw.npz (raw hit grid)
    - static_map.npz (final walls)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import os

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from roboracer_interfaces.msg import WaypointArray

from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix
from rclpy.time import Time

from roboracer_utils.frenet_converter import FrenetConverter

from ament_index_python.packages import get_package_share_directory


class BuildStaticMap(Node):
    def __init__(self):
        super().__init__("build_static_map")

        # Parameters
        self.declare_parameter("hits_output", "static_hits_raw.npz")
        self.declare_parameter("map_output", "static_map.npz")
        self.declare_parameter("s_res", 0.05)
        self.declare_parameter("d_res", 0.05)
        self.declare_parameter("d_min", -4.0) 
        self.declare_parameter("d_max", 4.0) 
        self.declare_parameter("min_hits", 5)   # min hit count for a cell to be considered stable

        hits_output_name = self.get_parameter("hits_output").get_parameter_value().string_value
        map_output_name = self.get_parameter("map_output").get_parameter_value().string_value
        self.s_res = self.get_parameter("s_res").get_parameter_value().double_value
        self.d_res = self.get_parameter("d_res").get_parameter_value().double_value
        self.d_min = self.get_parameter("d_min").get_parameter_value().double_value
        self.d_max = self.get_parameter("d_max").get_parameter_value().double_value
        self.min_hits = self.get_parameter("min_hits").get_parameter_value().integer_value
        self.hits_output = '/home/lyh/ros2_ws/src/f110_gym/localization/static_map/' + hits_output_name
        self.map_output = '/home/lyh/ros2_ws/src/f110_gym/localization/static_map/' + map_output_name

        # Frenet / track info
        self.converter = None
        self.track_length = None

        # Static hits grid
        self.static_hits = None     # shape (Ns, Nd), uint16 counts
        self.s_axis = None          # shape (Ns,), s coordinates
        self.d_bins = None          # shape (Nd,), d bin centers

        # For building H_map_bl from odom and TF (laser -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.H_map_bl = None

        self.wall_s = []
        self.wall_d = []

        # Subscribers
        self.create_subscription(WaypointArray, "/global_centerline", self.path_callback, 10)
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        self.frame_count = 0

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def path_callback(self, msg: WaypointArray):
        """Initialize Frenet converter and static hit grid from global_centerline."""
        if len(msg.wpnts) == 0:
            self.get_logger().warn("Received empty global_centerline.")
            return

        xs, ys, ss = [], [], []
        for wp in msg.wpnts:
            xs.append(float(wp.x_m))
            ys.append(float(wp.y_m))
            ss.append(float(wp.s_m))

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        ss = np.array(ss, dtype=np.float64)

        self.converter = FrenetConverter(xs, ys)
        self.track_length = float(ss[-1])  # 31.41
        self.get_logger().info(
            f"track_length={self.track_length:.2f} m."
        )

        Ns = int(self.track_length / self.s_res) + 1
        Nd = int((self.d_max - self.d_min) / self.d_res) + 1

        self.static_hits = np.zeros((Ns, Nd), dtype=np.uint16)
        self.s_axis = np.arange(Ns, dtype=np.float64) * self.s_res
        self.d_bins = self.d_min + (np.arange(Nd, dtype=np.float64) + 0.5) * self.d_res

        self.get_logger().info(
            f"[build_static_map] Initialized Frenet grid: Ns={Ns}, Nd={Nd}, "
            f"s_res={self.s_res:.3f}, d_res={self.d_res:.3f}, track_length={self.track_length:.2f}"
        )

    def odom_callback(self, odom: Odometry):
        """Build H_map_bl assuming map and odom are the same frame."""
        if self.static_hits is None:
            return

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation

        # Homogeneous transform map <- base_link
        H = quaternion_matrix([q.x, q.y, q.z, q.w])
        H[0, 3] = p.x
        H[1, 3] = p.y
        H[2, 3] = p.z
        self.H_map_bl = H

    def scan_callback(self, scan: LaserScan):
        """Accumulate LiDAR hits in static_hits grid."""
        if self.converter is None or self.static_hits is None or self.track_length is None:
            return
        if self.H_map_bl is None:
            return

        # 1) Laser -> base_link (use TF)
        pts_bl = self.laser_to_base_link(scan)
        if pts_bl is None or pts_bl.shape[0] == 0:
            return

        # 2) base_link -> map using H_map_bl
        pts_map = self.transform_xy(pts_bl, self.H_map_bl)
        if pts_map.size == 0:
            return

        # 3) Accumulate hits in Frenet grid
        self.accumulate_hits(pts_map)

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"[build_static_map] Accumulated {self.frame_count} scan frames."
            )

    # ---------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------
    def laser_to_base_link(self, scan: LaserScan):
        """Transform LaserScan points into ego_racecar/base_link using TF."""
        N = len(scan.ranges)
        if N == 0:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(N, dtype=np.float64) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]
        if ranges.size == 0:
            return None

        x_l = ranges * np.cos(angles)
        y_l = ranges * np.sin(angles)
        pts_l = np.stack([x_l, y_l], axis=1)

        laser_frame = scan.header.frame_id or "ego_racecar/laser_model"

        try:
            # base_link <- laser_frame
            tf_bl_from_l = self.tf_buffer.lookup_transform(
                "ego_racecar/base_link",
                laser_frame,
                Time(),  # latest available
                rclpy.duration.Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"[build_static_map] TF lookup failed base_link<-{laser_frame}: {e}")
            return None

        H_bl_l = self.tf_to_matrix(tf_bl_from_l)
        pts_bl = self.transform_xy(pts_l, H_bl_l)
        return pts_bl

    def tf_to_matrix(self, tf_msg):
        """Convert geometry_msgs/TransformStamped to 4x4 matrix."""
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        H = quaternion_matrix([q.x, q.y, q.z, q.w])
        H[0, 3] = t.x
        H[1, 3] = t.y
        H[2, 3] = t.z
        return H

    def transform_xy(self, pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply 4x4 transform to 2D XY points (z=0)."""
        if pts_xy.size == 0:
            return pts_xy
        N = pts_xy.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.hstack([pts_xy, np.zeros((N, 1)), ones])
        out = (H @ pts_h.T).T
        return out[:, :2].astype(np.float64)

    # ---------------------------------------------------------
    # Accumulate & build static walls
    # ---------------------------------------------------------
    def accumulate_hits(self, pts_map: np.ndarray):
        """Accumulate hits into static_hits in Frenet(s, d) space."""
        x = pts_map[:, 0].astype(np.float64)
        y = pts_map[:, 1].astype(np.float64)

        s_arr, d_arr = self.converter.get_frenet(x, y)
        s_arr = np.mod(s_arr, self.track_length)  # wrap s to [0, track_length)

        d_cut = 3.0 
        near_mask = np.abs(d_arr) < d_cut
        self.wall_s.append(s_arr[near_mask])
        self.wall_d.append(d_arr[near_mask])

    def save_all(self):
        """Save raw hits and final static wall map."""
        if self.static_hits is None or self.s_axis is None or self.d_bins is None:
            self.get_logger().warn("[build_static_map] static_hits not initialized, nothing to save.")
            return
        
        if self.wall_s:
            all_s = np.concatenate(self.wall_s)
            all_d = np.concatenate(self.wall_d)
        else:
            all_s = np.array([], dtype=np.float64)
            all_d = np.array([], dtype=np.float64)

        np.savez(self.hits_output,
                track_length=self.track_length,
                s_res=self.s_res,
                wall_s=all_s,
                wall_d=all_d)


def main(args=None):
    rclpy.init(args=args)
    node = BuildStaticMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, saving static map...")
        node.save_all()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
