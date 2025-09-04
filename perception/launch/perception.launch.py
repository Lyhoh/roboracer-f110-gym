from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter

def generate_launch_description():
    return LaunchDescription([

        # SetParameter(name='use_sim_time', value=True),

        Node(
            package='perception',
            executable='tf_bridge_map_odom',  
            name='tf_bridge_map_odom',
            output='screen',
        ),
        Node(
            package='perception',
            executable='detect',  
            name='detect',
            output='screen',
        ),
    ])
