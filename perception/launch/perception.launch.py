from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter

def generate_launch_description():
    return LaunchDescription([
        # SetParameter(name='use_sim_time', value=True),
        Node(
            package='perception',
            executable='waypoints_from_csv',  
            name='waypoints_from_csv',
            output='screen',
        ),
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
        Node(
            package='perception',
            executable='opponent_tracker',  
            name='opponent_tracker',
            output='screen',
        ),
    ])
