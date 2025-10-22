from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    right_controller = ExecuteProcess(
        cmd=[
        'gnome-terminal', '--', 'bash', '-c',
        'ros2 run adlap_tool_control controller --ros-args -r __ns:=/right --log-level debug; exec bash'

        ],
        output='screen',
    )

    return LaunchDescription([
        right_controller,
    ])