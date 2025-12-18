from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter

def generate_launch_description():
    return LaunchDescription([
        # SetParameter(name='use_sim_time', value=True),
        Node(
            package='localization',
            executable='build_static_map',  
            name='build_static_map',
            output='screen',
        ),
    ])
