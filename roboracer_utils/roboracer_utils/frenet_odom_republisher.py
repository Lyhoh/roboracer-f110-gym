#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from roboracer_interfaces.msg import WaypointArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from roboracer_utils.frenet_converter import FrenetConverter


class FrenetOdomRepublisher(Node):
    """
    This node converts /odom in Cartesian coordinates into /odom_frenet
    (s, d, v_s, v_d) using your FrenetConverter implementation.

    Input:
      - /global_centerline : f110_msgs/WaypointArray  (centerline of the track)
      - /odom             : nav_msgs/Odometry   (x, y, yaw, vx, vy) in map frame

    Output:
      - /odom_frenet      : nav_msgs/Odometry
            pose.position.x = s
            pose.position.y = d
            twist.linear.x  = v_s (if psi available, otherwise use vx)
            twist.linear.y  = v_d (if psi available, otherwise 0.0)
    """

    def __init__(self) -> None:
        super().__init__("frenet_odom_republisher")

        self.converter: Optional[FrenetConverter] = None
        self.track_ready = False
        self.path_needs_update = True 

        # Subscribe to global waypoints
        qos_wp = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,   
        )
        self.create_subscription(
            WaypointArray,
            "/global_centerline",        
            self.global_waypoints_cb,
            qos_wp,
        )

        # Subscribe to raw odom in Cartesian
        self.create_subscription(
            Odometry,
            "/ego_racecar/odom",  
            self.odom_cb,                   
            20,
        )

        # Publish Frenet odom
        self.pub_frenet_odom = self.create_publisher(
            Odometry,
            "/ego_frenet",
            10,
        )

        self.get_logger().info("[FrenetOdomRepublisher] Node initialized.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def global_waypoints_cb(self, msg: WaypointArray) -> None:
        """
        Get global centerline waypoints and initialize FrenetConverter.
        """
        if not self.path_needs_update:
            return
    
        if len(msg.wpnts) == 0:
            self.get_logger().warn("[FrenetOdomRepublisher] Received empty global_waypoints.")
            return

        xs = [wp.x_m for wp in msg.wpnts]
        ys = [wp.y_m for wp in msg.wpnts]

        self.converter = FrenetConverter(np.array(xs, dtype=np.float64),
                                         np.array(ys, dtype=np.float64))

        self.track_ready = True
        self.path_needs_update = False
        self.get_logger().info(
            f"[FrenetOdomRepublisher] Got {len(xs)} global waypoints, FrenetConverter initialized."
        )

    def odom_cb(self, msg: Odometry) -> None:
        """
        Convert incoming Cartesian odom into Frenet odom and republish.
        """
        if not self.track_ready or self.converter is None:
            # Still waiting for global_waypoints
            self.get_logger().warn("[FrenetOdomRepublisher] No global waypoints yet; cannot convert odom.")
            return

        # --- Extract Cartesian state from /odom ---
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # --- Use your FrenetConverter ---
        try:
            # get_frenet expects numpy arrays; it returns shape (2, N)
            xy_s = self.converter.get_frenet(
                np.array([x], dtype=np.float64),
                np.array([y], dtype=np.float64)
            )
            s_array, d_array = xy_s
            s = float(s_array[0])
            d = float(d_array[0])
        except Exception as e:
            self.get_logger().warn(f"[FrenetOdomRepublisher] get_frenet error: {e}")
            return

        # Compute Frenet velocities if waypoints_psi is available,
        # otherwise fall back to vx as v_s and 0.0 as v_d.
        if self.converter.waypoints_psi is not None:
            try:
                vs_vd = self.converter.get_frenet_velocities(vx, vy, yaw, s)
                v_s = float(vs_vd[0])
                v_d = float(vs_vd[1])
            except Exception as e:
                self.get_logger().warn(f"[FrenetOdomRepublisher] get_frenet_velocities error: {e}")
                v_s = float(vx)
                v_d = 0.0
        else:
            v_s = float(vx)
            v_d = 0.0

        # --- Build Frenet odom message ---
        frenet_msg = Odometry()
        frenet_msg.header = msg.header  

        frenet_msg.pose.pose.position.x = s
        frenet_msg.pose.pose.position.y = d

        frenet_msg.twist.twist.linear.x = v_s
        frenet_msg.twist.twist.linear.y = v_d

        frenet_msg.pose.pose.orientation = msg.pose.pose.orientation
        frenet_msg.twist.twist.angular = msg.twist.twist.angular

        self.pub_frenet_odom.publish(frenet_msg)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """
        Convert quaternion to yaw angle (in radians).
        """
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrenetOdomRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
