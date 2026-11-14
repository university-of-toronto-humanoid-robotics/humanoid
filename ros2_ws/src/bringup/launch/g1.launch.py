"""The whole G1 walking stack, on the real robot or in the sim.

  ros2 launch bringup g1.launch.py                                  # sim, scenario room_walk
  ros2 launch bringup g1.launch.py scenario:=room_stairs_heightmap scan:=elevation_map
  ros2 launch bringup g1.launch.py sim:=false policy_type:=rsl_rl_g1_heightmap checkpoint:=/path/model.pt

sim picks what plays the hardware, and nothing else:
  sim:=true    mujoco_g1_walk_sim: MuJoCo's motors, pelvis IMU and Mid-360
  sim:=false   the real G1 (it publishes /lowstate and obeys /lowcmd by itself) and the Livox driver

Everything above the hardware is the same either way: joint_state_bridge (/lowstate to /joint_states),
perception.launch.py (FAST-LIO, the elevation map, /height_scan) and walk_controller (the policy).

The controller does nothing until told to. In the sim the scenario does that; on the robot, with it hanging from
the gantry and still:
  ros2 service call /walk_controller/start std_srvs/srv/Trigger     # stand, then walk
  ros2 service call /walk_controller/stop std_srvs/srv/Trigger      # go limp
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

SCAN_TOPICS = {'elevation_map': '/height_scan', 'true_terrain': '/ground_truth/height_scan'}


def setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    sim = arg('sim').lower() == 'true'
    policy_type, checkpoint, scan = arg('policy_type'), arg('checkpoint'), arg('scan')
    if sim:
        # The scenario names the policy that suits its terrain; the arguments override it.
        from bridge.scenario import in_repo, load_scenario
        policy = load_scenario(arg('scenario'))['policy']
        policy_type = policy_type or policy['type']
        checkpoint = checkpoint or in_repo(policy['checkpoint'])
        scan = scan or policy.get('scan', 'elevation_map')
    else:
        scan = scan or 'elevation_map'
        if not policy_type or not checkpoint:
            raise RuntimeError('On the real robot, pass policy_type:= and checkpoint:=')
        if scan != 'elevation_map':
            raise RuntimeError('The true terrain only exists in the sim: use scan:=elevation_map')
    if scan not in SCAN_TOPICS:
        raise RuntimeError(f'Unknown scan "{scan}": use elevation_map or true_terrain')
    time = {'use_sim_time': sim}

    if sim:
        hardware = [Node(
            package='bridge',
            executable='mujoco_g1_walk_sim',
            output='screen',
            parameters=[{'scenario': arg('scenario'), 'viewer': arg('viewer').lower() == 'true',
                         'hold': float(arg('hold'))}],
            on_exit=Shutdown(reason='the scenario ended'),
        )]
    else:
        hardware = [IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('livox_ros_driver2'), 'launch_ROS2', 'msg_MID360_launch.py')))]

    return hardware + [
        Node(package='bridge', executable='joint_state_bridge', parameters=[time]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('bringup'), 'launch', 'perception.launch.py')),
            launch_arguments={'use_sim_time': str(sim).lower(), 'rviz': arg('rviz'),
                              'lidar_upside_down': 'true'}.items(),
        ),
        Node(
            package='policy',
            executable='walk_controller',
            output='screen',
            parameters=[time, {'policy_type': policy_type, 'checkpoint': checkpoint,
                               'scan_topic': SCAN_TOPICS[scan]}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true',
                              description='true: MuJoCo plays the robot. false: the real G1'),
        DeclareLaunchArgument('scenario', default_value='room_walk',
                              description='Sim only: a name from sim/scenarios, or a path to a scenario file'),
        DeclareLaunchArgument('policy_type', default_value='',
                              description='rsl_rl_g1_flat or rsl_rl_g1_heightmap. Sim: defaults to the scenario\'s'),
        DeclareLaunchArgument('checkpoint', default_value='',
                              description='Path to the policy checkpoint. Sim: defaults to the scenario\'s'),
        DeclareLaunchArgument('scan', default_value='',
                              description='Where a height-map policy\'s terrain scan comes from: elevation_map '
                                          '(as on the robot) or, sim only, true_terrain'),
        DeclareLaunchArgument('hold', default_value='3.0',
                              description='Sim only: s the motion script waits once the controller is walking'),
        DeclareLaunchArgument('viewer', default_value='true', description='Sim only: open the MuJoCo viewer'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        OpaqueFunction(function=setup),
    ])
