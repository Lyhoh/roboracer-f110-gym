import os
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('controller_py')
    param_file = os.path.join(pkg_share, 'config', 'pure_pursuit.yaml')

    return LaunchDescription([
        # SetParameter(name='use_sim_time', value=True),
        Node(
            package='controller_py',
            executable='pure_pursuit',
            name='pure_pursuit',
            output='screen',
            parameters=[param_file], 
        )
    ])
