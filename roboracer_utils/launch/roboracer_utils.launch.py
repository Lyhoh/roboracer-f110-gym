from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        Node(
            package='roboracer_utils',
            executable='tf_bridge_map_odom',  
            name='tf_bridge_map_odom',
            output='screen',
        ),

        Node(
            package='roboracer_utils',
            executable='frenet_odom_republisher',  
            name='frenet_odom_republisher',
            output='screen',
        ),
    ])
