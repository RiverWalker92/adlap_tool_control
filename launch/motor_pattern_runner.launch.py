from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.actions import EmitEvent
from launch_ros.actions import Node


# No Digital Twin is launched during the motor trial.
# The Motor DT is evaluated offline after data collection to avoid 
#prediction-induced timing delays.

def generate_launch_description():
    # Output directory and log file name for the ROS data
    output_dir = LaunchConfiguration("output_dir")
    log_file_name = LaunchConfiguration("log_file_name")

    # Generates and publishes the motor command pattern.
    motor_pattern_runner_node = Node(
        package="adlap_tool_control",
        executable="motor_pattern_runner_node.py",
        name="motor_pattern_runner_node",
        output="screen",
    )

    # Records the ROS data generated during the motor-only trial.
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

    # Delay the motor pattern briefly so the data logger can initialize first.
    delayed_motor_pattern_runner = TimerAction(
        period=3.0,
        actions=[motor_pattern_runner_node],
    )

    # Shut down the complete launch when the pattern has finished.
    shutdown_when_pattern_is_done = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_pattern_runner_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(
                        reason="Motor pattern runner finished"
                    )
                )
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("log_file_name"),

        trial_logger_node,
        delayed_motor_pattern_runner,
        shutdown_when_pattern_is_done,
    ])