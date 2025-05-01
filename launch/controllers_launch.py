from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    right_controller = Node(
        package="adlap_tool_control",
        executable="controller",
        name="right_tool_controller",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"my_parameter": "right"}
        ]
    )

    left_controller = Node(
        package="adlap_tool_control",
        executable="controller",
        name="left_tool_controller",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"my_parameter": "left"}
        ]
    )

    # Add the nodes to the launch description
    return LaunchDescription([
        right_controller,
        left_controller
    ])