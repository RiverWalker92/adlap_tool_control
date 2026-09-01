from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch_ros.actions import Node

# Configuration files for the nodes
def generate_launch_description():

    # Parameters for the DOF pattern runner node
    params_file = (
        Path.home()
        / "ros2_ws"
        / "src"
        / "adlap_tool_control"
        / "config"
        / "dof_pattern_params.yaml"
    )

    # Parameters for the instrument state node
    instrument_params_file = (
        Path.home()
        / "ros2_ws"
        / "src"
        / "adlap_tool_control"
        / "config"
        / "instrument_state_node.yaml"
    )

    # Output directory and log file name for the ROS data
    output_dir = LaunchConfiguration("output_dir")
    log_file_name = LaunchConfiguration("log_file_name")

    # Hardware configuration detected before starting the trial.
    #
    # coupling_mode:
    #   motor_only   -> Motor DT only
    #   gearbox_only -> Motor DT + Gearbox DT
    #   full_setup   -> Motor DT + Gearbox DT + Instrument DT
    coupling_mode = LaunchConfiguration("coupling_mode")
    active_dof = LaunchConfiguration("active_dof")

    # Selects the gearbox and instrument configuration
    gearbox_variant = LaunchConfiguration("gearbox_variant")
    instrument_config = LaunchConfiguration("instrument_config")

    # Executes the DOF pattern runner node with the specified parameters and configuration.
    dof_pattern_runner_node = Node(
        package="adlap_tool_control",
        executable="dof_pattern_runner_node.py",
        name="dof_pattern_runner_node",
        parameters=[
            str(params_file),
            {
                "coupling_mode": coupling_mode,
                "active_dof": active_dof,
            },
        ],
        output="screen",
    )

    # Records the ROS data generated during the trials.
    trial_logger_node = Node(
        package="adlap_tool_control",
        executable="trial_logger_node.py",
        output="screen",
        parameters=[
            {
                "output_dir": output_dir,
                "log_file_name": log_file_name,
            }
        ],
    )

    # Executes the gearbox state node with the specified gearbox variant configuration.
    gearbox_state_node = Node(
        package="adlap_tool_control",
        executable="gearbox_state_node.py",
        output="screen",
        parameters=[
            {
                "gearbox_variant": gearbox_variant,
            }
        ],
        condition=IfCondition(
            PythonExpression([
                "'", coupling_mode, "' == 'gearbox_only' or '", coupling_mode, "' == 'full_setup'"
            ])
        )
    )

    # The Instrument DT is only required when instrument is coupled and thus when the complete setup is used.
    instrument_state_node = Node(
        package="adlap_tool_control",
        executable="instrument_state_node.py",
        parameters=[
            str(instrument_params_file),
            {
                "instrument_config": instrument_config,
            }

        ],
        output="screen",
        condition=IfCondition(
            PythonExpression([
                "'", coupling_mode, "' == 'full_setup'"
            ])
        ),
    )

    # Shut down the complete launch when the pattern has finished.
    shutdown_when_pattern_is_done = RegisterEventHandler(
        OnProcessExit(
            target_action=dof_pattern_runner_node,
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
            "coupling_mode",
            description="Detected coupling mode: motor_only, gearbox_only or full_setup.",
        ),

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

        DeclareLaunchArgument(
            "active_dof",
            default_value="0",
            description="DOF to run as sequence: 1, 2, 3 or 4. Use 0 for YAML-defined behavior.",
        ),

        dof_pattern_runner_node,
        trial_logger_node,
        gearbox_state_node,
        instrument_state_node,
        shutdown_when_pattern_is_done,
    ])