import csv
import os
from typing import List, Tuple
import math
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Header

from roboracer_interfaces.msg import Waypoint, WaypointArray
from roboracer_utils.frenet_converter import FrenetConverter


class GlobalWaypointsFromCsvNode(Node):
    """
    A single ROS2 node that:
      1) Reads the track CENTERLINE CSV file
      2) Reads the RACELINE CSV file
      3) Converts raceline (x,y) into Frenet (s,d) using the centerline as reference
      4) Publishes:
            /global_centerline   (Frenet reference + track boundaries)
            /global_raceline     (optimal path in same Frenet frame)
    """

    def __init__(self):
        super().__init__("global_waypoints_from_csv")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("map_name", "Austin")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate", 0.5)  # Hz, 0 → publish once

        map_name = self.get_parameter("map_name").value
        self.frame_id = self.get_parameter("frame_id").value
        rate = float(self.get_parameter("publish_rate").value)

        pkg_share = get_package_share_directory("f1tenth_gym_ros")
        tracks_dir = os.path.join(pkg_share, "maps", "f1tenth_racetracks")

        center_csv = os.path.join(tracks_dir, map_name, f"{map_name}_centerline.csv")
        race_csv = os.path.join(tracks_dir, map_name, f"{map_name}_raceline.csv")

        # ----------------------------
        # QoS: transient local → late subscribers also receive last message
        # ----------------------------
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub_center = self.create_publisher(WaypointArray, "/global_centerline", qos)
        self.pub_race = self.create_publisher(WaypointArray, "/global_raceline", qos)
        self.pub_marker = self.create_publisher(Marker, "/planner/global_waypoints_markers", 1)

        # ----------------------------
        # Load centerline
        # ----------------------------
        (
            self.center_s,
            self.center_x,
            self.center_y,
            self.center_dl,
            self.center_dr,
            self.centerline_msg,
        ) = self.load_centerline(center_csv)

        # Convert centerline arrays to FrenetConverter
        self.converter = FrenetConverter(
            np.array(self.center_x, dtype=np.float64),
            np.array(self.center_y, dtype=np.float64),
        )

        # ----------------------------
        # Load raceline (requires converter)
        # ----------------------------
        self.raceline_msg = self.load_raceline(race_csv)

        # ----------------------------
        # Initial publish
        # ----------------------------
        self.publish_once()

        self.publish_waypoint_points(self.raceline_msg)

        # Optional periodic publishing
        if rate > 0.0:
            self.timer = self.create_timer(1.0 / rate, self.publish_once)
        else:
            self.timer = None

        self.get_logger().info(
            f"[GLOBAL WPs] Loaded {len(self.centerline_msg.wpnts)} centerline points"
        )
        self.get_logger().info(
            f"[GLOBAL WPs] Loaded {len(self.raceline_msg.wpnts)} raceline points"
        )

    def yaw_to_quat(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        q.x = 0.0
        q.y = 0.0
        return q
    
    def load_centerline(
        self, path: str
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], WaypointArray]:
        """
        Reads a centerline CSV containing:
            x_m, y_m, w_tr_right_m, w_tr_left_m
        Computes cumulative arc length s_m and returns:
            s_list, x_list, y_list, d_left_list, d_right_list, WaypointArray
        """

        # Lists for output
        s_list: List[float] = []
        x_list: List[float] = []
        y_list: List[float] = []
        dl_list: List[float] = []
        dr_list: List[float] = []

        # ----- 1) Read CSV (new format) -----
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)          # first line is header
            header = [h.strip() for h in header]

            for row in reader:
                # Skip empty rows
                if not row:
                    continue
                # Skip comment-only rows like "# something"
                if row[0].startswith("#") and len(row) == 1:
                    continue
                # Convert all fields to float
                rows.append([float(x) for x in row])

        if not rows:
            raise RuntimeError(f"Centerline file '{path}' has no data rows.")

        data = np.array(rows, dtype=np.float64)

        # Map column names to indices (header may have "#x_m" or "x_m")
        col = {name: i for i, name in enumerate(header)}
        x_col = col["# x_m"] if "# x_m" in col else col["x_m"]
        y_col = col["y_m"]
        wr_col = col["w_tr_right_m"]
        wl_col = col["w_tr_left_m"]

        x_arr = data[:, x_col]
        y_arr = data[:, y_col]
        # Track widths: positive to the right / left of centerline
        w_tr_right = data[:, wr_col]
        w_tr_left  = data[:, wl_col]

        # ----- 2) Compute cumulative arc length s_m -----
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        seg_len = np.hypot(dx, dy)              # segment lengths
        s_arr = np.concatenate([[0.0], np.cumsum(seg_len)])  # s[0]=0, then cumulative

        # In Frenet-style naming: d_left/d_right are just the widths
        d_left_arr  = w_tr_left
        d_right_arr = w_tr_right

        # Convert to Python lists
        s_list = s_arr.tolist()
        x_list = x_arr.tolist()
        y_list = y_arr.tolist()
        dl_list = d_left_arr.tolist()
        dr_list = d_right_arr.tolist()

        # ----- 3) Fill WaypointArray message -----
        msg = WaypointArray()
        msg.header = Header(frame_id=self.frame_id)

        wp_list: List[Waypoint] = []
        for i, (s, x, y, dl, dr) in enumerate(
            zip(s_list, x_list, y_list, dl_list, dr_list)
        ):
            wp = Waypoint()
            wp.id = i

            # Centerline has d_m = 0
            wp.s_m = s
            wp.d_m = 0.0

            wp.x_m = x
            wp.y_m = y

            # d_left / d_right are positive widths from centerline to walls
            wp.d_left = dl
            wp.d_right = dr

            # Centerline has no curvature/speed info here
            wp.psi_rad = 0.0
            wp.kappa_radpm = 0.0
            wp.vx_mps = 0.0
            wp.ax_mps2 = 0.0

            wp_list.append(wp)

        msg.wpnts = wp_list
        return s_list, x_list, y_list, dl_list, dr_list, msg

    # ============================================================
    # Load raceline CSV (convert to centerline Frenet frame)
    # ============================================================
    def load_raceline(self, path: str) -> WaypointArray:
        """
        Reads raceline CSV of format:
            s; x; y; psi; kappa; vx; ax
        Then:
            - Projects (x,y) to centerline Frenet → get (s_m, d_m)
            - Interpolates centerline (d_left, d_right) at this s_m
        Returns a WaypointArray.
        """
        wp_list = []

        center_s = np.array(self.center_s, dtype=np.float64)
        center_dl = np.array(self.center_dl, dtype=np.float64)
        center_dr = np.array(self.center_dr, dtype=np.float64)

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p for p in line.split(";") if p != ""]
                if len(parts) < 7:
                    continue

                # Raceline original fields
                s_raw = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                psi = float(parts[3])
                kappa = float(parts[4])
                vx = float(parts[5])
                ax = float(parts[6])

                # Project raceline (x,y) onto centerline Frenet
                s_r_arr, d_arr = self.converter.get_frenet(
                    np.array([x], dtype=np.float64),
                    np.array([y], dtype=np.float64),
                )
                s_ref = float(s_r_arr[0])   # longitudinal coordinate in centerline frame
                d_m = float(d_arr[0])       # lateral offset from centerline

                # Interpolate centerline track width at s_ref
                dl = float(np.interp(s_ref, center_s, center_dl))
                dr = float(np.interp(s_ref, center_s, center_dr))

                wp = Waypoint()
                wp.id = len(wp_list)
                wp.s_m = s_ref
                wp.d_m = d_m
                wp.x_m = x
                wp.y_m = y
                wp.d_left = dl
                wp.d_right = dr
                wp.psi_rad = psi
                wp.kappa_radpm = kappa
                wp.vx_mps = vx
                wp.ax_mps2 = ax

                wp_list.append(wp)

        msg = WaypointArray()
        msg.header = Header(frame_id=self.frame_id)
        msg.wpnts = wp_list
        return msg

    # ============================================================
    # Publish both centerline + raceline
    # ============================================================
    def publish_once(self):
        now = self.get_clock().now().to_msg()
        self.centerline_msg.header.stamp = now
        self.raceline_msg.header.stamp = now
        self.pub_center.publish(self.centerline_msg)
        self.pub_race.publish(self.raceline_msg)

    def publish_waypoint_points(self, msg: WaypointArray):
        marker = Marker()
        marker.header = msg.header
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS          
        marker.action = Marker.ADD

        marker.scale.x = 0.1
        marker.scale.y = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0  # no additional rotation, identity

        for wp in msg.wpnts:
            p = Point()
            p.x = wp.x_m
            p.y = wp.y_m
            p.z = 0.0
            marker.points.append(p)

        self.pub_marker.publish(marker)



def main():
    rclpy.init()
    node = GlobalWaypointsFromCsvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
