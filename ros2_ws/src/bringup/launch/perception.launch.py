import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('bringup')
    fast_lio_dir = get_package_share_directory('fast_lio')
    cupy_dir = get_package_share_directory('elevation_mapping_cupy')

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    lidar_upside_down = LaunchConfiguration('lidar_upside_down')

    """
        gravity_align node first aligns odom with gravity by reading the livox accelerometer at startup, then 
        publishes a transform from odom->camera_init to connect the FAST-LIO TF tree (camera_init->body) with
        odom for navigation purposes
        - camera_init is the Mid-360 IMU's pose when FAST-LIO starts, so it inherits the mount orientation
            and any tilt of the torso at that moment. odom is the z-up world frame: gravity_align levels it
            with the accelerometer while the robot stands still at startup, keeping the torso's heading.
        - On the G1, odom's origin is torso_link at startup, using the torso_link -> mid360_imu pose from
            description/g1/g1_29dof_rev_1_0.urdf. odom -> camera_init appears about 1 s after the IMU starts.
        - Also accounts for the IMU->radar offset inside the mid360, which isn't given in the robot urdf, but in mid360.yaml:
            - extrinsic_T [ -0.011, -0.02329, 0.04412 ]: radar relative to IMU.
            - Using SE(3) transform to get precise odom->LiDAR IMU, this works out to be [0.0135, -0.0233, 0.4719]
    """

    # When we use the robot (LiDAR upside down)
    odom_to_camera_init_g1 = Node(
        package='bridge',
        executable='gravity_align',
        parameters=[{'x': 0.0135, 'y': -0.0233, 'z': 0.4719,
                     'roll': 3.141592653589793, 'pitch': 0.05112069379091391, 'yaw': 0.0,
                     'use_sim_time': use_sim_time}],
        condition=IfCondition(lidar_upside_down),
    )
    
    # When we just use the LiDAR (no robot, LiDAR is upright)
    odom_to_camera_init_upright = Node(
        package='bridge',
        executable='gravity_align',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(lidar_upside_down),
    )


    # The G1 model. RViz can't resolve the URDF's relative mesh paths, so point them at the installed copy.
    with open(os.path.join(bringup_dir, 'description', 'g1', 'g1_29dof_rev_1_0.urdf')) as urdf:
        robot_description = urdf.read().replace(
            'filename="meshes/', 'filename="package://bringup/description/g1/meshes/')
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': use_sim_time}],
        condition=IfCondition(lidar_upside_down),
    )

    # Joins FAST-LIO's tree to the URDF's: odom -> camera_init -> body -> pelvis -> ...
    body_to_pelvis = Node(
        package='bridge',
        executable='body_to_pelvis',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(lidar_upside_down),
    )

    # A Node rather than an include of fast_lio's mapping.launch.py: includes aren't scoped, so its
    # rviz argument would overwrite ours, and its arguments would show up in `ros2 launch -s`.
    fast_lio = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        output='screen',
        parameters=[
            os.path.join(fast_lio_dir, 'config', 'mid360.yaml'),
            {'use_sim_time': use_sim_time},
        ],
    )

    # Elevation mapping node
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
        odom_to_camera_init_upright,
        robot_state_publisher,
        body_to_pelvis,
        fast_lio,
        elevation_mapping,
        rviz_node,
    ])
