from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # f1tenth_dir = PathJoinSubstitution([FindPackageShare('f1tenth_gym_ros'), 'launch'])
    # perception_dir = PathJoinSubstitution([FindPackageShare('perception'), 'launch'])
    # control_dir = PathJoinSubstitution([FindPackageShare('controller_py'), 'launch'])
    # util_dir = PathJoinSubstitution([FindPackageShare('utils'), 'launch'])
    # global_planner_dir = PathJoinSubstitution([FindPackageShare('global_planner'), 'launch'])
    # local_planner_dir = PathJoinSubstitution([FindPackageShare('local_planner'), 'launch'])
    # return LaunchDescription([
    #     IncludeLaunchDescription(
    #         PathJoinSubstitution([f1tenth_dir, 'gym_bridge_launch.py'])
    #     ),
    #     IncludeLaunchDescription(
    #         PathJoinSubstitution([perception_dir, 'perception.launch.py'])
    #     ),
    #     IncludeLaunchDescription(
    #         PathJoinSubstitution([control_dir, 'pure_pursuit.launch.py'])
    #     ),
    #     IncludeLaunchDescription(
    #         PathJoinSubstitution([util_dir, 'utils.launch.py'])
    #     ),
    #     IncludeLaunchDescription(
    #         PathJoinSubstitution([global_planner_dir, 'global_planner.launch.py'])
    #     ),
    #     # IncludeLaunchDescription(
    #     #     PathJoinSubstitution([local_planner_dir, 'local_planner.launch.py'])
    #     # ),
    # ])

    map_name = LaunchConfiguration('map_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name',
            default_value='IMS',   
            description='racetrack / map name'
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py']),
            launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('perception'), 'launch', 'perception.launch.py']),
            # launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(PathJoinSubstitution([FindPackageShare('controller_py'), 'launch', 'pure_pursuit.launch.py']),
            # launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('utils'), 'launch', 'utils.launch.py']),
            # launch_arguments={'map_name': map_name}.items()
        ),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('global_planner'), 'launch', 'global_planner.launch.py']),
            launch_arguments={'map_name': map_name}.items()
        ),

        # IncludeLaunchDescription(
        #     PathJoinSubstitution([FindPackageShare('local_planner'), 'launch', 'local_planner.launch.py']),
        #     # launch_arguments={'map_name': map_name}.items()
        # ),
    ])