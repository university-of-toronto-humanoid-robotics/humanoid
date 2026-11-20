"""The operator's camera window: the G1's head camera, the only view of the robot
the operator gets during a trial. Run keyboard_teleop in a second terminal (it
needs the keyboard, so it can't start from here)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from launch_ros.actions import Node


def setup(context):
    image = context.launch_configurations['image']
    transport = context.launch_configurations['transport']
    if transport == 'raw':
        return [Node(package='rqt_image_view', executable='rqt_image_view', arguments=[image])]
    # Decompress here, on the operator's machine, so only the compressed stream crosses Wi-Fi. rqt_image_view can't be
    # given image_raw/compressed itself: it subscribes before discovery has listed the topic, takes the name for a raw
    # Image topic, and crashes on the type clash when it later sees the real CompressedImage topic.
    local = '/operator/head_camera/image_raw'
    return [
        Node(
            package='image_transport',
            executable='republish',
            name='operator_decompress',
            parameters=[{'in_transport': transport, 'out_transport': 'raw'}],
            remappings=[(f'in/{transport}', f'{image}/{transport}'), ('out', local)],
        ),
        Node(package='rqt_image_view', executable='rqt_image_view', arguments=[local]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image', default_value='/g1/head_camera/color/image_raw',
                              description='Camera topic to show (the raw base topic)'),
        # Compressed by default: raw video is too much for Wi-Fi. transport:=raw shows the raw stream.
        DeclareLaunchArgument('transport', default_value='compressed',
                              description='image_transport to receive it over: compressed or raw'),
        OpaqueFunction(function=setup),
    ])
