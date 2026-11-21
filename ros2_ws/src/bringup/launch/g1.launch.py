"""The whole G1 walking stack, on the real robot or in the sim.

  ros2 launch bringup g1.launch.py                                  # sim, scenario room_walk
  ros2 launch bringup g1.launch.py scenario:=room_stairs_heightmap scan:=elevation_map
  ros2 launch bringup g1.launch.py sim:=false policy_type:=rsl_rl_g1_heightmap checkpoint:=/path/model.pt

sim picks what plays the hardware, and nothing else:
  sim:=true    mujoco_g1_walk_sim: MuJoCo's motors, pelvis IMU and Mid-360
  sim:=false   the real G1 (it publishes /lowstate and obeys /lowcmd by itself) and the Livox driver

Everything above the hardware is the same either way: joint_state_bridge (/lowstate to /joint_states),
perception.launch.py (FAST-LIO, the elevation map, /height_scan), walk_controller (the policy) and, on the
competition course, the localizer (map -> odom: where the robot is on the course).

The localizer needs to know roughly where the pelvis starts: initial_pose:="x, y, yaw_deg" in the course frame
(sim/competition/README.md), within about 0.5 m and 20 deg. In the sim it runs by itself on course scenes and
starts from the scenario's spawn; on the robot it runs when initial_pose is given:
  ros2 launch bringup g1.launch.py sim:=false policy_type:=... checkpoint:=... initial_pose:="10.8, 4.2, 180"

With the localizer runs the navigator, which walks a route (route:=name, from ros2_ws/src/navigation/routes) or to
a /goal_pose (RViz's "2D Goal Pose"). It waits to be told:
  ros2 service call /navigator/start std_srvs/srv/Trigger    # walk the route
  ros2 service call /navigator/stop std_srvs/srv/Trigger     # stand still where it is
In the sim, a scenario with motion type navigator names the route and starts it.

The controller does nothing until told to. In the sim the scenario does that; on the robot, with it hanging from
the gantry and still:
  ros2 service call /walk_controller/start std_srvs/srv/Trigger     # stand, then walk
  ros2 service call /walk_controller/stop std_srvs/srv/Trigger      # go limp
"""
import math
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from bridge.scenario import in_repo, load_scenario

SCAN_TOPICS = {'elevation_map': '/height_scan', 'true_terrain': '/ground_truth/height_scan'}


def setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    sim = arg('sim').lower() == 'true'
    policy_type, checkpoint, scan = arg('policy_type'), arg('checkpoint'), arg('scan')
    allow_reset = False   # walk_controller's ~/reset: only for sim testing, never on the robot
    if sim:
        # The scenario names the policy that suits its terrain; the arguments override it.
        scenario = load_scenario(arg('scenario'))
        policy = scenario['policy']
        allow_reset = scenario.get('on_fall', 'end') == 'respawn'
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

    # Localization: on course scenes in the sim, from the spawn; on the robot, when initial_pose is given.
    localize, initial = arg('localize').lower(), arg('initial_pose')
    if sim:
        spawn = scenario.get('spawn', {})
        on_course = 'course' in Path(scenario['scene']).stem
        initial = initial or f"{spawn.get('x', 0.0)}, {spawn.get('y', 0.0)}, {math.degrees(spawn.get('yaw', 0.0))}"
        localize = localize or ('true' if on_course else 'false')
    else:
        localize = localize or ('true' if initial else 'false')
    route = arg('route')
    if sim:
        motion = scenario.get('motion', {})
        route = route or (motion.get('route', '') if motion.get('type') == 'navigator' else '')
    if route and not route.endswith('.yaml'):
        route = in_repo(f'ros2_ws/src/navigation/routes/{route}.yaml')
    course_nodes = []   # the localizer and the navigator
    if localize == 'true':
        if not initial:
            raise RuntimeError('localize:=true needs initial_pose:="x, y, yaw_deg" (where the pelvis starts)')
        course_nodes = [Node(
            package='localization',
            executable='localizer',
            output='screen',
            parameters=[time, {'map_file': in_repo('sim/competition/course_heightmap.npz'),
                               'initial_pose': [float(v) for v in initial.split(',')]}],
        ), Node(
            package='navigation',
            executable='navigator',
            output='screen',
            parameters=[time, {'route_file': route}],
        )]
    elif route:
        raise RuntimeError('route:= needs the localizer: turn it on with localize:=true and initial_pose:=')

    if sim:
        hardware = [Node(
            package='bridge',
            executable='mujoco_g1_walk_sim',
            output='screen',
            parameters=[{'scenario': arg('scenario'), 'viewer': arg('viewer').lower() == 'true',
                         'hold': float(arg('hold'))}],
            on_exit=Shutdown(reason='the scenario ended'),
        ), Node(
            # The head camera's compressed (JPEG) stream for the operator, as the RealSense driver adds on the robot.
            package='image_transport',
            executable='republish',
            parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
            remappings=[('in', '/g1/head_camera/color/image_raw'),
                        ('out/compressed', '/g1/head_camera/color/image_raw/compressed')],
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
                               'scan_topic': SCAN_TOPICS[scan], 'allow_reset': allow_reset}],
        ),
        # The operator's status line (ros2_ws/src/teleop): tilt and measured speed, from /lowstate and FAST-LIO.
        Node(package='teleop', executable='robot_status', parameters=[time]),
        # One driver at a time: teleop (/cmd_vel/teleop) or the navigator (/cmd_vel/nav) to /cmd_vel. The operator
        # wins: touching a stick stops the navigator.
        Node(package='navigation', executable='cmd_vel_mux', parameters=[time]),
    ] + course_nodes


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
        DeclareLaunchArgument('localize', default_value='',
                              description='true or false: run the localizer. Default: sim on course scenes; robot '
                                          'when initial_pose is given'),
        DeclareLaunchArgument('route', default_value='',
                              description='A route for the navigator: a name from ros2_ws/src/navigation/routes, or '
                                          'a path. Sim: defaults to the scenario\'s'),
        DeclareLaunchArgument('initial_pose', default_value='',
                              description='"x, y, yaw_deg": where the pelvis starts on the course, roughly. '
                                          'Sim: defaults to the scenario\'s spawn'),
        OpaqueFunction(function=setup),
    ])
