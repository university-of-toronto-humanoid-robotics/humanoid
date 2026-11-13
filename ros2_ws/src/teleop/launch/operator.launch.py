"""The operator's camera window: the G1's head camera, the only view of the robot
the operator gets during a trial. Run keyboard_teleop in a second terminal (it
needs the keyboard, so it can't start from here)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image', default_value='/g1/head_camera/image_raw',
                              description='Camera topic to show'),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=[LaunchConfiguration('image')],
        ),
    ])
