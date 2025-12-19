from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    map_name = LaunchConfiguration('map_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='Austin',   
            description='racetrack / map name'
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py']),
            launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('global_planner'), 'launch', 'global_planner.launch.py']),
            launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('localization'), 'launch', 'build_static_map.launch.py']),
        ),

        IncludeLaunchDescription(PathJoinSubstitution([FindPackageShare('controller'), 'launch', 'pure_pursuit.launch.py']),
        ),
    ])