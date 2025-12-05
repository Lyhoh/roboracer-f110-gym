import csv
from typing import List
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from interfaces.msg import Waypoint, WaypointArray  # adjust package name if different

class WaypointsFromCsvNode(Node):
    def __init__(self):
        super().__init__('waypoints_from_csv')

        # --- Parameters ---
        self.declare_parameter('csv_path', '/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline.csv')
        self.declare_parameter('topic', '/global_centerline')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 0.5)  # Hz; 0 -> publish once then stop  # 0.5

        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.topic = self.get_parameter('topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        rate = float(self.get_parameter('publish_rate').value)

        if not csv_path:
            self.get_logger().error('Parameter "csv_path" is empty.')
            raise SystemExit(1)

        # --- QoS: transient_local so late subscribers get the last message ---
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(WaypointArray, self.topic, qos)

        # Load CSV into WaypointsArray
        self.msg = self.load_csv_as_msg(csv_path)

        # Publish once immediately
        self.publish_once()

        # Optionally republish at a low rate (useful if some nodes join very late without transient)
        if rate > 0.0:
            self.timer = self.create_timer(1.0 / rate, self.publish_once)
        else:
            self.timer = None

        self.get_logger().info(f'[WP CSV] Loaded {len(self.msg.wpnts)} waypoints from {csv_path}')
        self.get_logger().info(f'[WP CSV] Publishing on {self.topic} with frame_id="{self.frame_id}"')

    def load_csv_as_msg(self, path: str) -> WaypointArray:
        wp_list: List[Waypoint] = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or row.get('x_m', '').startswith('#'):
                    continue
                x = float(row['x_m']); y = float(row['y_m']); s = float(row['s_m'])
                dl = float(row['d_left']); dr = float(row['d_right'])
                wp = Waypoint(x_m=x, y_m=y, s_m=s, d_left=dl, d_right=dr)
                wp_list.append(wp)

        # Optional: sanity checks
        if len(wp_list) < 2:
            self.get_logger().warn('Waypoint count < 2; check CSV.')
        if not all(wp_list[i].s_m <= wp_list[i+1].s_m for i in range(len(wp_list)-1)):
            self.get_logger().warn('s_m is not non-decreasing. Some consumers may assume sorted s_m.')

        msg = WaypointArray()
        msg.header = Header()
        msg.wpnts = wp_list
        return msg

    def publish_once(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.header.frame_id = self.frame_id
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    node = WaypointsFromCsvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
