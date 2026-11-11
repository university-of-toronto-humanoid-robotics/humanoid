import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('bringup')
    fast_lio_dir = get_package_share_directory('fast_lio')
    cupy_dir = get_package_share_directory('elevation_mapping_cupy')

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    lidar_upside_down = LaunchConfiguration('lidar_upside_down')

    # camera_init is the LiDAR's pose at startup, so it inherits the mount orientation.
    # odom is the same place but z-up: on the G1 it's the torso at startup, using the
    # torso_link -> mid360_link joint from description/g1/g1_29dof_rev_1_0.urdf.
    # Assumes the robot is standing upright when FAST-LIO starts.
    odom_to_camera_init_g1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0.0002835', '--y', '0.00003', '--z', '0.428434',
                   '--roll', '3.141592653589793', '--pitch', '0.05112069379091391', '--yaw', '0',
                   '--frame-id', 'odom', '--child-frame-id', 'camera_init'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(lidar_upside_down),
    )
    # Upright LiDAR (e.g. ~/rosbag_01): camera_init is already z-up.
    odom_to_camera_init_identity = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--frame-id', 'odom', '--child-frame-id', 'camera_init'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(lidar_upside_down),
    )

    fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(fast_lio_dir, 'launch', 'mapping.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time, 'rviz': 'false'}.items(),
    )

    elevation_mapping = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            os.path.join(cupy_dir, 'config', 'core', 'core_param.yaml'),
            os.path.join(bringup_dir, 'config', 'g1_elevation_mapping.yaml'),
            {'use_sim_time': use_sim_time},
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(bringup_dir, 'rviz', 'perception.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use the bag clock (set true when playing a rosbag)'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('lidar_upside_down', default_value='true',
                              description='true on the G1 (flipped Mid-360); false for upright-LiDAR bags'),
        odom_to_camera_init_g1,
        odom_to_camera_init_identity,
        fast_lio,
        elevation_mapping,
        rviz_node,
    ])
