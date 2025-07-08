from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    left_controller = ExecuteProcess(
        cmd=[
        'gnome-terminal', '--', 'bash', '-c',
        'ros2 run adlap_tool_control controller --ros-args -r __ns:=/left; exec bash'

        ],
        output='screen',
    )

    return LaunchDescription([
        left_controller,
    ])