import os
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('controller')
    param_file = os.path.join(pkg_share, 'config', 'pure_pursuit.yaml')

    return LaunchDescription([
        Node(
            package='controller',
            executable='pp_overtake',
            name='pp_overtake',
            output='screen',
            parameters=[param_file], 
        ),
    ])
