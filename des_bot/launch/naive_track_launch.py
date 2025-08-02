from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='des_bot',
            namespace='des_bot_0',
            executable='vision_pub',
            name='vision_pub'
        ),
        Node(
            package='des_bot',
            namespace='des_bot_0',
            executable='naive_track',
            name='naive_track'
        )
    ])