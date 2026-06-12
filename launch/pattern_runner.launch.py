from pathlib import Path

from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
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

    instrument_params_file = (
        Path.home()
        / "ros2_ws"
        / "src"
        / "adlap_tool_control"
        / "config"
        / "instrument_state_node.yaml"
    )

    pattern_runner_node = Node(
        package="adlap_tool_control",
        executable="pattern_runner_node.py",
        name="tool_controller_node",   # belangrijk: moet matchen met tool_params.yaml
        parameters=[str(params_file)],
        output="screen",
    )

    tool_reader_node = Node(
        package="adlap_tool_control",
        executable="tool_reader_node.py",
        output="screen",
    )

    gearbox_state_node = Node(
        package="adlap_tool_control",
        executable="gearbox_state_node.py",
        output="screen",
    )

    instrument_state_node = Node(
        package="adlap_tool_control",
        executable="instrument_state_node.py",
        parameters=[str(instrument_params_file)],
        output="screen",
    )

    shutdown_when_pattern_is_done = RegisterEventHandler(
        OnProcessExit(
            target_action=pattern_runner_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="Pattern runner finished")
                )
            ],
        )
    )

    return LaunchDescription([
        pattern_runner_node,
        tool_reader_node,
        gearbox_state_node,
        instrument_state_node,
        shutdown_when_pattern_is_done,
    ])