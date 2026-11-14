"""The G1's hardware in MuJoCo: its motors, pelvis IMU and Mid-360, in a scene picked by a scenario file.

It stands in for the robot and nothing else. Like the real G1 it obeys /lowcmd and publishes /lowstate (Unitree's
own messages), /livox/lidar and /livox/imu. Everything above that is the same software as on the robot, started
by bringup's g1.launch.py: the walking policy runs in walk_controller, /joint_states comes from
joint_state_bridge, and the terrain scan from height_scan_node.

The robot starts hanging from a gantry, as the real one does: these gains can't balance it standing still, only
the policy can. Until a controller sends /lowcmd the sim holds the joints in the standing pose.

A scenario (sim/scenarios/*.yaml) picks the scene, the spawn pose and a motion script (bridge/motion.py). The sim
also plays the operator, doing what a person does with the real robot: it starts the controller once FAST-LIO is
running, lets go of the gantry once the controller reports that the policy is walking, and then publishes the
motion script's /cmd_vel, steering by the true pose. The robot is built the way mjlab built the one the policies
were trained on (bridge/g1_model.py).

For testing only, it also publishes /clock, /ground_truth/odom and /ground_truth/height_scan (the true terrain
scan), and at the end reports FAST-LIO's error and how far /height_scan was from the true scan.
"""
from pathlib import Path
import threading
import time

from geometry_msgs.msg import Twist
from livox_ros_driver2.msg import CustomMsg
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from bridge import g1_model
from bridge.height_scan import ScanMonitor, TrueTerrainScan
from bridge.motion import make_motion
from bridge.mujoco_mid360_sim import FRAME_SLICES, IMU_DT, SLICE_TICKS, STAMP_OFFSET, Mid360Sim, to_stamp
from bridge.scenario import in_repo, load_scenario
from policy import g1, lowcmd

SETTLE = 4.0                # s on the gantry before sim time starts: the torso takes about 3 s to stop swaying
FALLEN_HEIGHT = 0.45        # m, pelvis height above the ground below it, under which the robot has fallen
OPERATOR_TICKS = 4          # the operator's /cmd_vel and the true scan go out every 4th physics step: 50 Hz
START_AFTER_ODOMETRY = 1.0  # s the operator waits after FAST-LIO's first pose before starting the controller
MODE_MACHINE = 5            # what a 29-joint G1 reports as its type in /lowstate
N = len(g1.JOINTS)


class G1WalkSim(Node):

    # The LiDAR, its IMU and the FAST-LIO error report are the scripted sim's. They use self.data, rng, gyro,
    # accel, lidar, rays, ray_index, rays_per_slice, the noise settings, gt, t_start, errors and the publishers.
    sensor = Mid360Sim.sensor
    publish_imu = Mid360Sim.publish_imu
    record_ground_truth = Mid360Sim.record_ground_truth
    declare_lidar_noise = Mid360Sim.declare_lidar_noise
    add_lidar_noise = Mid360Sim.add_lidar_noise
    trace_slice = Mid360Sim.trace_slice
    on_odometry = Mid360Sim.on_odometry
    report = Mid360Sim.report

    def __init__(self):
        super().__init__('mujoco_g1_walk_sim')
        self.scenario_name = self.declare_parameter('scenario', 'room_walk').value
        scenario = load_scenario(self.scenario_name)
        # Overrides of the scenario; empty or 0 keeps the scenario's value.
        scene = self.declare_parameter('scene', '').value or in_repo(scenario['scene'])
        self.duration = self.declare_parameter('duration', 0.0).value or float(scenario['duration'])
        speed = self.declare_parameter('speed', 0.0).value
        description = self.declare_parameter('description', in_repo('description/g1/g1_29dof_rev_1_0.xml')).value
        self.viewer = self.declare_parameter('viewer', True).value
        # s the operator waits, once the controller is walking, before commanding any motion.
        self.hold = self.declare_parameter('hold', 3.0).value
        self.rays_per_slice = self.declare_parameter('rays_per_slice', 2000).value
        seed = self.declare_parameter('seed', 0).value
        self.rng = np.random.default_rng(seed)
        self.declare_lidar_noise(seed)

        self.model = g1_model.build_model(scene, description)
        self.data = mujoco.MjData(self.model)
        self.motion = make_motion(scenario['motion'], speed)
        s = scenario.get('spawn', {})
        ground = g1_model.spawn(self.model, self.data, s.get('x', 0.0), s.get('y', 0.0), s.get('yaw', 0.0))
        self.data.ctrl[:] = g1.default_pose()   # the joints hold the standing pose until a controller takes over
        g1_model.hold_on_gantry(self.model, self.data)
        self.get_logger().info(f'Scenario "{self.scenario_name}": {Path(scene).name}, spawn at ({s.get("x", 0.0)}, '
                               f'{s.get("y", 0.0)}) on ground {ground:.3f} m')

        from mujoco_lidar import MjLidarWrapper
        from mujoco_lidar.scan_gen import LivoxGenerator
        # Trace the scene and the robot's visual meshes (groups 0-2), not its collision capsules (group 3).
        # torso_link is left out because the mid360 site is inside its head mesh.
        self.lidar = MjLidarWrapper(
            self.model, 'mid360', backend='warp', cutoff_dist=40.0,
            args={'bodyexclude': self.model.body('torso_link').id, 'geomgroup': np.array([1, 1, 1, 0, 0, 0])})
        self.rays = LivoxGenerator('mid360').ray_angles.astype(np.float32)
        self.ray_index = 0

        self.gyro = self.model.sensor('mid360-imu-angular-velocity')
        self.accel = self.model.sensor('mid360-imu-linear-acceleration')
        self.pelvis_accel = self.model.sensor('imu_lin_acc')
        hinges = np.nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.joint_qpos = self.model.jnt_qposadr[hinges]
        self.joint_dof = self.model.jnt_dofadr[hinges]

        # The robot's interface.
        self.lowstate_pub = self.create_publisher(LowState, '/lowstate', 10)
        self.imu_pub = self.create_publisher(Imu, '/livox/imu', 200)
        self.lidar_pub = self.create_publisher(CustomMsg, '/livox/lidar', 10)
        self.create_subscription(LowCmd, '/lowcmd', self.on_lowcmd, 10)
        self.lowstate = LowState()
        self.lowstate.mode_machine = MODE_MACHINE
        self.lowcmd = None          # the newest command not yet applied
        self.commanded = False      # whether a controller has taken over

        # The operator, and the test instruments.
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gt_pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.true_scan_pub = self.create_publisher(Float32MultiArray, '/ground_truth/height_scan',
                                                   qos_profile_sensor_data)
        self.create_subscription(Odometry, '/Odometry', self.on_odometry, 100)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/walk_controller/state', self.on_controller_state, latched)
        self.start_client = self.create_client(Trigger, '/walk_controller/start')
        self.start_sent = False
        self.true_scan = TrueTerrainScan(self.model, self.data)
        self.monitor = ScanMonitor(self)
        self.controller_state = None
        self.motion_start = None    # sim time at which the operator starts the motion script

        self.gt = []
        self.t_start = None
        self.errors = []

    def on_lowcmd(self, msg):
        # Runs on the ROS thread. The physics loop picks the command up, so the model never changes mid-step.
        if msg.crc != lowcmd.crc(msg):
            self.get_logger().error('Dropped a /lowcmd with a wrong crc, as the robot would.',
                                    throttle_duration_sec=2.0)
            return
        self.lowcmd = msg

    def on_controller_state(self, msg):
        self.controller_state = msg.data

    def apply_lowcmd(self):
        msg, self.lowcmd = self.lowcmd, None
        if msg is None:
            return
        if not self.commanded:
            self.commanded = True
            self.get_logger().info('A controller is sending /lowcmd: the motors are its now.')
        motors = msg.motor_cmd[:N]
        on = np.array([m.mode != 0 for m in motors])    # a motor switched off gives no torque
        g1_model.set_motor_commands(
            self.model, self.data,
            q=np.array([m.q for m in motors]), dq=np.array([m.dq for m in motors]),
            tau=np.array([m.tau for m in motors]) * on,
            kp=np.array([m.kp for m in motors]) * on, kd=np.array([m.kd for m in motors]) * on)

    def publish_lowstate(self, tick):
        msg = self.lowstate
        msg.tick = int(tick * IMU_DT * 1000)   # ms
        w, x, y, z = self.data.qpos[3:7]
        imu = msg.imu_state
        imu.quaternion = np.array([w, x, y, z], dtype=np.float32)
        imu.gyroscope = self.data.qvel[3:6].astype(np.float32)   # a free joint's turn rate is in its body's frame
        imu.accelerometer = self.sensor(self.pelvis_accel).astype(np.float32)
        imu.rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz').astype(np.float32)
        q, dq, torque = self.data.qpos[self.joint_qpos], self.data.qvel[self.joint_dof], self.data.actuator_force
        for i in range(N):
            motor = msg.motor_state[i]
            motor.mode, motor.q, motor.dq, motor.tau_est = 1, float(q[i]), float(dq[i]), float(torque[i])
        self.lowstate_pub.publish(msg)

    def pelvis_pose(self):
        w, x, y, z = self.data.qpos[3:7]
        return self.data.qpos[:3], Rotation.from_quat([x, y, z, w]).as_euler('ZYX')[0]

    def height_above_ground(self):
        pos = self.data.qpos[:3]
        ground = g1_model.ground_below(self.model, self.data, pos[0], pos[1], top=pos[2])
        return np.inf if ground is None else pos[2] - ground

    @property
    def moving(self):
        """Whether the motion script has a start time: the controller is walking."""
        return self.motion_start is not None

    def operate(self, t):
        """The operator's part, and the true terrain scan for comparison."""
        if (not self.start_sent and self.errors and t >= self.errors[0][0] + START_AFTER_ODOMETRY
                and self.start_client.service_is_ready()):
            self.start_sent = True
            self.start_client.call_async(Trigger.Request())
            self.get_logger().info(f'FAST-LIO is running: starting the controller at t = {t:.1f} s.')
        if not self.moving and self.controller_state == 'walk':
            g1_model.release_gantry(self.data)
            self.motion_start = t + self.hold
            self.get_logger().info(f'The controller is walking at t = {t:.1f} s: gantry released. Motion starts in '
                                   f'{self.hold:.0f} s.')
        command = np.zeros(3)
        if self.moving and t >= self.motion_start:
            command = self.motion.command(t - self.motion_start, *self.pelvis_pose())
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = map(float, command)
        self.cmd_vel_pub.publish(msg)

        truth = self.true_scan(self.data)
        self.true_scan_pub.publish(Float32MultiArray(data=truth.tolist()))
        self.monitor.record(truth)

    def run(self):
        for _ in range(int(SETTLE / IMU_DT)):
            mujoco.mj_step(self.model, self.data)
        torso = self.data.body('torso_link')
        rot = Rotation.from_matrix(torso.xmat.reshape(3, 3))
        roll, pitch = rot.as_euler('xyz', degrees=True)[:2]
        heading = rot.as_euler('ZYX', degrees=True)[0]
        x, y, z = torso.xpos
        self.get_logger().info(
            f'On the gantry after {SETTLE:.0f} s: pelvis {self.height_above_ground():.3f} m above the ground, torso '
            f'roll {roll:+.2f} pitch {pitch:+.2f} deg (gravity_align levels odom for this). odom will be about torso_link '
            f'now, levelled: ({x:.3f}, {y:.3f}, {z:.3f}) m in the scene, heading {heading:.2f} deg')

        self.get_logger().info('Waiting for a /livox/lidar subscriber (start it all with bringup g1.launch.py)...')
        while rclpy.ok() and self.count_subscribers('/livox/lidar') == 0:
            time.sleep(0.1)
        self.get_logger().info(f'Running "{self.scenario_name}" for up to {self.duration:.0f} s of sim time.')

        viewer = None
        if self.viewer:
            from mujoco import viewer as mujoco_viewer
            viewer = mujoco_viewer.launch_passive(self.model, self.data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.model.body('pelvis').id
            viewer.cam.distance = 4.0

        wall_start = time.monotonic()
        points, frame_start, slices, tick = [], 0.0, 0, 0
        fell, finished = False, None
        while rclpy.ok() and tick * IMU_DT < self.duration and (viewer is None or viewer.is_running()):
            t = tick * IMU_DT
            ahead = wall_start + t - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)

            self.apply_lowcmd()
            clock = Clock()
            clock.clock = to_stamp(t)
            self.clock_pub.publish(clock)
            operator_tick = tick % OPERATOR_TICKS == 0
            if operator_tick:
                self.operate(t)
            # After mj_step, positions and sensors are still those at t; only qpos and qvel have moved on.
            mujoco.mj_step(self.model, self.data)
            self.publish_lowstate(tick)
            self.publish_imu(t)
            self.record_ground_truth(t)

            if tick % SLICE_TICKS == 0:
                if slices == FRAME_SLICES:
                    msg = CustomMsg()
                    msg.header.stamp = to_stamp(frame_start)
                    msg.header.frame_id = 'livox_frame'
                    msg.timebase = int(round((frame_start + STAMP_OFFSET) * 1e9))
                    msg.point_num = len(points)
                    msg.points = points
                    self.lidar_pub.publish(msg)
                    points, slices, frame_start = [], 0, t
                offset_ns = int(round((t - frame_start) * 1e9))
                points += self.trace_slice(offset_ns, slices * self.rays_per_slice)
                slices += 1

            if operator_tick and self.height_above_ground() < FALLEN_HEIGHT:
                fell = True
                self.get_logger().error(f'The robot fell at t = {t:.1f} s, near ({self.data.qpos[0]:.2f}, '
                                        f'{self.data.qpos[1]:.2f}), after {self.motion.progress}.')
                break
            if not self.start_sent and tick == int(10.0 / IMU_DT):
                self.get_logger().warn('Still on the gantry after 10 s: the controller is only started once FAST-LIO '
                                       'publishes /Odometry and /walk_controller/start exists. Are both running?')
            if finished is None and self.moving and t >= self.motion_start and self.motion.done:
                finished = t
                self.get_logger().info(f'Motion finished at t = {t:.1f} s.')
            if finished is not None and t > finished + 2.0:
                break
            if viewer is not None and tick % 6 == 0:
                viewer.sync()
            tick += 1

        time.sleep(2.0)   # give FAST-LIO time to finish its last frames
        realtime = tick * IMU_DT / max(time.monotonic() - wall_start - 2.0, 1e-3)
        if not fell:
            self.get_logger().info(f'Completed {self.motion.progress}. Ran at {realtime:.2f}x real time.')
        self.report()
        self.get_logger().info(self.monitor.report())
        if viewer is not None:
            viewer.close()


def main():
    rclpy.init()
    node = G1WalkSim()
    # ROS callbacks run on their own thread, so a command is never stuck behind a physics step.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinning = threading.Thread(target=executor.spin, daemon=True)
    spinning.start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.report()
    finally:
        executor.shutdown()
        spinning.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
