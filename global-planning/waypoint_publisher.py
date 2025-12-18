#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from roboracer_interfaces.msg import Waypoint, WaypointArray

import numpy as np
import csv
import os

class WaypointPublisher(Node):
    """
    Publish /global_centerline from a CSV.
    CSV must be either:
      A) x_m,y_m,s_m,d_left,d_right  (preferred)
      B) x_m,y_m,w_tr_right_m,w_tr_left_m (then we compute s_m)
    """

    def __init__(self):
        super().__init__('waypoint_publisher')

        # params
        self.declare_parameter('csv_path', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)

        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        rate = float(self.get_parameter('publish_rate_hz').get_parameter_value().double_value)

        if not csv_path or not os.path.isfile(csv_path):
            raise FileNotFoundError(f'CSV not found: {csv_path}')

        # load CSV
        data, header = self._load_csv(csv_path)
        cols = {name: header.index(name) for name in header}

        # figure out mode
        has_s = 's_m' in cols
        has_bounds = ('d_left' in cols and 'd_right' in cols)
        has_widths = ('w_tr_left_m' in cols and 'w_tr_right_m' in cols)

        if not (has_bounds or has_widths):
            raise RuntimeError('CSV must have d_left+d_right OR w_tr_left_m+w_tr_right_m columns.')

        # build message
        waypoints = []
        # compute s if needed
        if not has_s:
            xy = data[:, [cols['#x_m'] if '#x_m' in cols else cols['x_m'], cols['y_m']]]
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            s_m = np.concatenate([[0.0], np.cumsum(seg)])
        else:
            s_m = data[:, cols['s_m']]

        x_col = cols['#x_m'] if '#x_m' in cols else cols['x_m']
        y_col = cols['y_m']

        if has_bounds:
            d_left = data[:, cols['d_left']]
            d_right = data[:, cols['d_right']]
        else:
            # convert widths to d_left/d_right naming
            d_left  = data[:, cols['w_tr_left_m']]
            d_right = data[:, cols['w_tr_right_m']]

        for i in range(data.shape[0]):
            wp = Waypoint()
            wp.x_m = float(data[i, x_col])
            wp.y_m = float(data[i, y_col])
            wp.s_m = float(s_m[i])
            wp.d_left = float(d_left[i])
            wp.d_right = float(d_right[i])
            waypoints.append(wp)

        self.msg = WaypointArray()
        self.msg.header = Header()
        self.msg.header.frame_id = frame_id
        self.msg.wpnts = waypoints

        self.pub = self.create_publisher(WaypointArray, '/global_centerline', 10)
        self.timer = self.create_timer(1.0/max(rate, 1e-3), self._tick)
        self.get_logger().info(f'Loaded {len(waypoints)} waypoints from {csv_path}')

    def _tick(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)

    @staticmethod
    def _load_csv(path):
        # returns (np.array, header_list)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                rows.append([float(x) for x in row])
        return np.array(rows, dtype=np.float32), [h.strip() for h in header]

def main():
    rclpy.init()
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
