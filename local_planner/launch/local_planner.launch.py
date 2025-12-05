from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter

def generate_launch_description():
    return LaunchDescription([
        # SetParameter(name='use_sim_time', value=True),
        Node(
            package='local_planner',
            executable='local_planner',  
            name='local_planner',
            output='screen',
        ),
    ])
