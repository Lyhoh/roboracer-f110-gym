from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    map_name = LaunchConfiguration('map_name')

    return LaunchDescription([
        DeclareLaunchArgument('map_name', default_value='Austin'),

        Node(
            package='global_planner',
            executable='waypoints_from_csv',  
            name='waypoints_from_csv',
            parameters=[{
                "map_name": map_name,
            }],
            output="screen",
        ),
    ])
