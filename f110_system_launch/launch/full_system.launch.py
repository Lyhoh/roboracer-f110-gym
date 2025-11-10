from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    f1tenth_dir = PathJoinSubstitution([FindPackageShare('f1tenth_gym_ros'), 'launch'])
    perception_dir = PathJoinSubstitution([FindPackageShare('perception'), 'launch'])
    control_dir = PathJoinSubstitution([FindPackageShare('controller_py'), 'launch'])
    return LaunchDescription([
        IncludeLaunchDescription(
            PathJoinSubstitution([f1tenth_dir, 'gym_bridge_launch.py'])
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([perception_dir, 'perception.launch.py'])
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution([control_dir, 'pure_pursuit.launch.py'])
        ),
    ])