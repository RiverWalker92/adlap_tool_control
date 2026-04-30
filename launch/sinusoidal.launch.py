from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_name = 'adlap_tool_control'

    default_params_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'sinusoidal_params.yaml'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to parameter file'
    )

    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='/right/tool_control_node/instrument_angles',
        description='Output topic'
    )

    node = Node(
        package=pkg_name,
        executable='sinusoidal_controller',
        name='sinusoidal_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'topic': LaunchConfiguration('topic')
            }
        ]
    )

    return LaunchDescription([
        params_file_arg,
        topic_arg,
        node
    ])