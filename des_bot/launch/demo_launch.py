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
            executable='conversation_manager',
            name='conversation_manager',
        ),
        Node(
            package='des_bot',
            namespace='des_bot_0',
            executable='expression_manager',
            name='expression_manager',
        ),
       
        Node(
            package='des_bot',
            namespace='des_bot_0',
            executable='button_handler',
            name='button_handler',
        )
    ])