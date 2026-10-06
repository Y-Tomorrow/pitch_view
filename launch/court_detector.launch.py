import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('pitch_view'),
        'config',
        'court_detector.yaml',
    )

    return LaunchDescription([
        Node(
            package='pitch_view',
            executable='court_detector',
            name='court_detector',
            parameters=[config],
            output='screen',
        ),
    ])
