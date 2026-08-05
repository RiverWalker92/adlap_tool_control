from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir")
    log_file_name = LaunchConfiguration("log_file_name")

    dof_reader_node = Node(
        package="adlap_tool_control",
        executable="dof_reader_node.py",
        output="screen",
        parameters=[
            {
                "output_dir": output_dir,
                "log_file_name": log_file_name,
            }
        ],
    )

    motor_pattern_runner_node = Node(
        package="adlap_tool_control",
        executable="motor_pattern_runner_node.py",
        name="motor_pattern_runner_node",
        output="screen",
    )
    motor_dt_node = Node(
        package="adlap_tool_control",
        executable="motor_digital_twin.py",
        name="motor_digital_twin_node",
        output="screen",
    )

    delayed_motor_pattern_runner = TimerAction(
        period=3.0,
        actions=[motor_pattern_runner_node],
    )

    shutdown_when_pattern_is_done = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_pattern_runner_node,
            on_exit=[
                Shutdown(reason="Motor pattern runner finished")
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("log_file_name"),

        dof_reader_node,
        motor_dt_node,
        delayed_motor_pattern_runner,
        shutdown_when_pattern_is_done,
    ])