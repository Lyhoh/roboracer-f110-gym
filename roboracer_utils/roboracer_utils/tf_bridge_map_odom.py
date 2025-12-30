import argparse

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler


def publish_static_map_to_odom(node: Node,
                               map_frame: str,
                               odom_frame: str,
                               sx: float, sy: float, sz: float,
                               sroll: float, spitch: float, syaw: float):
    """Send a single static transform: map -> odom."""
    static_br = StaticTransformBroadcaster(node)
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = map_frame
    t.child_frame_id = odom_frame
    t.transform.translation.x = float(sx)
    t.transform.translation.y = float(sy)
    t.transform.translation.z = float(sz)
    q = quaternion_from_euler(float(sroll), float(spitch), float(syaw))
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    static_br.sendTransform(t)
    node.get_logger().info(
        f"[static] {map_frame} -> {odom_frame}  "
        f"xyz=({sx:.3f},{sy:.3f},{sz:.3f})  rpy=({sroll:.3f},{spitch:.3f},{syaw:.3f})"
    )
    return static_br  # keep reference alive


def main():
    # ---- CLI args ----
    parser = argparse.ArgumentParser(description="Bridge map->odom (static) and odom->base_link (dynamic TF).")
    parser.add_argument("--odom-topic", default="/ego_racecar/odom",
                        help="Odometry topic providing ego pose in odom frame.")
    parser.add_argument("--map-frame", default="map", help="Parent frame for the static transform.")
    parser.add_argument("--odom-frame", default="odom", help="Child frame of the static transform.")
    parser.add_argument("--base-link-frame", default="ego_racecar/base_link",
                        help="Child frame for dynamic TF from odom.")
    # Static transform parameters (map->odom)
    parser.add_argument("--static-x", type=float, default=0.0)
    parser.add_argument("--static-y", type=float, default=0.0)
    parser.add_argument("--static-z", type=float, default=0.0)
    parser.add_argument("--static-roll", type=float, default=0.0)
    parser.add_argument("--static-pitch", type=float, default=0.0)
    parser.add_argument("--static-yaw", type=float, default=0.0)
    args, _ = parser.parse_known_args()

    rclpy.init()

    # ---- Node ----
    node = Node("tf_bridge_map_odom")
    node.get_logger().info("Starting tf_bridge_map_odom")

    # ---- Publish static map->odom once (latched) ----
    static_br = publish_static_map_to_odom(
        node,
        args.map_frame, args.odom_frame,
        args.static_x, args.static_y, args.static_z,
        args.static_roll, args.static_pitch, args.static_yaw
    )

    # ---- Dynamic TF broadcaster for odom->base_link ----
    dyn_br = TransformBroadcaster(node)

    # ---- Odometry -> TF callback ----
    def odom_cb(msg: Odometry):
        # Build TF: odom -> base_link from Odometry pose
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = args.odom_frame
        t.child_frame_id = args.base_link_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        dyn_br.sendTransform(t)

    # ---- Subscriber ----
    node.create_subscription(Odometry, args.odom_topic, odom_cb, 10)
    node.get_logger().info(
        f"Bridging Odometry '{args.odom_topic}'  ==>  TF {args.odom_frame} -> {args.base_link_frame}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()