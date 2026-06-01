from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = (
        Path.home()
        / "ros2_ws"
        / "src"
        / "adlap_tool_control"
        / "config"
        / "tool_params.yaml"
    )

    return LaunchDescription([
        Node(
            package="adlap_tool_control",
            executable="pattern_runner_node.py",
            name="tool_controller_node",
            parameters=[str(params_file)],
            output="screen",
        ),

        Node(
            package="adlap_tool_control",
            executable="gearbox_state_node.py",
            output="screen",
        ),

        Node(
            package="adlap_tool_control",
            executable="tool_reader_node.py",
            output="screen",
        ),

        Node(
            package="adlap_tool_control",
            executable="pattern_runner_node.py",
            name="tool_controller_node",
            parameters=[str(params_file)],
            output="screen",
        ),
    ])