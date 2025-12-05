from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='utils',
            executable='frenet_odom_republisher',  
            name='frenet_odom_republisher',
            output='screen',
        ),
    ])
