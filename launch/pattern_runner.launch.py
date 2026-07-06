from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
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

    output_dir = LaunchConfiguration("output_dir")
    log_file_name = LaunchConfiguration("log_file_name")

    gearbox_variant = LaunchConfiguration("gearbox_variant")
    instrument_config = LaunchConfiguration("instrument_config")

    tool_pattern_runner_node = Node(
        package="adlap_tool_control",
        executable="tool_pattern_runner_node.py",
        name="tool_controller_node",   # belangrijk: moet matchen met tool_params.yaml
        parameters=[str(params_file)],
        output="screen",
    )

    tool_reader_node = Node(
        package="adlap_tool_control",
        executable="tool_reader_node.py",
        output="screen",
        parameters=[
            {
                "output_dir": output_dir,
                "log_file_name": log_file_name,
            }
        ],
    )

    gearbox_state_node = Node(
        package="adlap_tool_control",
        executable="gearbox_state_node.py",
        output="screen",
        parameters=[
            {
                "gearbox_variant": gearbox_variant,
            }
        ],
    )

    instrument_state_node = Node(
        package="adlap_tool_control",
        executable="instrument_state_node.py",
        parameters=[
            str(instrument_params_file),
            {
                "instrument_config": instrument_config,
                "hybrid_bend_enabled": True,
                "hybrid_bend_model_file": str(
                    Path.home()
                    / "ros2_ws"
                    / "test_data"
                    / "trained_models"
                    / "bend_prediction_gearbox_only_no_history"
                    / "gradient_boosting"
                    / "gradient_boosting_bend_hybrid_model.joblib"
                ),
            },
        ],
        output="screen",
    )
    shutdown_when_pattern_is_done = RegisterEventHandler(
        OnProcessExit(
            target_action=tool_pattern_runner_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="Pattern runner finished")
                )
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("log_file_name"),

        DeclareLaunchArgument(
            "gearbox_variant",
            default_value="",
            description="Gearbox variant selected by marker detection, e.g. gearbox_1 or gearbox_2.",
        ),

        DeclareLaunchArgument(
            "instrument_config",
            default_value="",
            description="Instrument config selected by marker detection, e.g. gripper.",
        ),

        tool_pattern_runner_node,
        tool_reader_node,
        gearbox_state_node,
        instrument_state_node,
        shutdown_when_pattern_is_done,
    ])