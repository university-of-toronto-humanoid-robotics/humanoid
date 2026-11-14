"""G1 walking in MuJoCo under a trained policy, with a simulated Mid-360, set up by a scenario file.

Like mujoco_mid360_sim, but with physics instead of a scripted pose. A scenario (sim/scenarios/*.yaml) picks the
scene, the spawn pose, the policy (bridge/walk_policy.py) and the motion script that commands it
(bridge/motion.py). The robot is built the way mjlab built the one the policy was trained on (bridge/g1_model.py).

Publishes the same topics as mujoco_mid360_sim (/clock, /livox/imu, /livox/lidar, /joint_states and
/ground_truth/odom) and reports FAST-LIO's error at the end. Motion scripts use the true pose.
"""
from pathlib import Path
import time

from livox_ros_driver2.msg import CustomMsg
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, JointState
import yaml

from bridge import g1_model
from bridge.height_scan import ElevationMapScan, TrueTerrainScan
from bridge.motion import make_motion
from bridge.mujoco_mid360_sim import FRAME_SLICES, IMU_DT, SLICE_TICKS, STAMP_OFFSET, Mid360Sim, to_stamp
from bridge.walk_policy import make_policy

SETTLE = 1.0                # s of standing before sim time starts, so the drop onto the ground isn't recorded
FALLEN_HEIGHT = 0.45        # m, pelvis height above the ground below it, under which the robot has fallen


def repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'sim' / 'scenarios').is_dir():
            return parent
    raise FileNotFoundError('No sim/scenarios above this file: build with --symlink-install')


def in_repo(path):
    path = Path(path)
    return str(path if path.is_absolute() else repo_root() / path)


def load_scenario(name):
    """A scenario by name (sim/scenarios/<name>.yaml) or by path."""
    path = Path(name) if name.endswith('.yaml') else repo_root() / 'sim' / 'scenarios' / f'{name}.yaml'
    if not path.is_file():
        known = ', '.join(p.stem for p in sorted((repo_root() / 'sim' / 'scenarios').glob('*.yaml')))
        raise FileNotFoundError(f'No scenario at "{path}". Known: {known}')
    with open(path) as f:
        return yaml.safe_load(f)


class G1WalkSim(Node):

    # The sensor publishing and error report are the scripted sim's. They use self.data, rng, gyro, accel,
    # lidar, rays, ray_index, rays_per_slice, the noise settings, joint_*, gt, t_start, errors and the publishers.
    sensor = Mid360Sim.sensor
    publish_imu = Mid360Sim.publish_imu
    publish_joints = Mid360Sim.publish_joints
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
        policy = self.declare_parameter('policy', '').value or in_repo(scenario['policy']['checkpoint'])
        self.duration = self.declare_parameter('duration', 0.0).value or float(scenario['duration'])
        speed = self.declare_parameter('speed', 0.0).value
        # Where a height-map policy's terrain scan comes from: true_terrain or elevation_map.
        self.scan_source = self.declare_parameter('scan', '').value or scenario['policy'].get('scan', 'true_terrain')
        description = self.declare_parameter('description', in_repo('description/g1/g1_29dof_rev_1_0.xml')).value
        self.viewer = self.declare_parameter('viewer', True).value
        self.hold = self.declare_parameter('hold', 3.0).value
        self.rays_per_slice = self.declare_parameter('rays_per_slice', 2000).value
        seed = self.declare_parameter('seed', 0).value
        self.rng = np.random.default_rng(seed)
        self.declare_lidar_noise(seed)
        if not Path(policy).is_file():
            raise FileNotFoundError(f'No walking policy at "{policy}". Pass -p policy:=<path to a checkpoint>')

        self.model = g1_model.build_model(scene, description)
        self.data = mujoco.MjData(self.model)
        self.scan = None
        if self.scan_source == 'true_terrain':
            self.scan = TrueTerrainScan(self.model, self.data)
        elif self.scan_source == 'elevation_map':
            self.scan = ElevationMapScan(self, TrueTerrainScan(self.model, self.data))
        else:
            raise ValueError(f'Unknown scan source "{self.scan_source}": use true_terrain or elevation_map')
        self.policy = make_policy(scenario['policy']['type'], policy, self.model, self.scan)
        if not self.policy.needs_scan:
            self.scan = None
        self.decimation = int(round(self.policy.dt / IMU_DT))
        self.motion = make_motion(scenario['motion'], speed)
        s = scenario.get('spawn', {})
        ground = g1_model.spawn(self.model, self.data, s.get('x', 0.0), s.get('y', 0.0), s.get('yaw', 0.0))
        self.get_logger().info(f'Scenario "{self.scenario_name}": {Path(scene).name}, spawn at ({s.get("x", 0.0)}, '
                               f'{s.get("y", 0.0)}) on ground {ground:.3f} m, {Path(policy).name}'
                               + (f', terrain scan from {self.scan_source}' if self.scan else ''))

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
        hinges = np.nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.joint_names = [self.model.joint(int(j)).name for j in hinges]
        self.joint_qpos = self.model.jnt_qposadr[hinges]
        self.joint_dof = self.model.jnt_dofadr[hinges]

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/livox/imu', 200)
        self.lidar_pub = self.create_publisher(CustomMsg, '/livox/lidar', 10)
        self.gt_pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.create_subscription(Odometry, '/Odometry', self.on_odometry, 100)

        self.gt = []
        self.t_start = None
        self.errors = []

    def pelvis_pose(self):
        w, x, y, z = self.data.qpos[3:7]
        return self.data.qpos[:3], Rotation.from_quat([x, y, z, w]).as_euler('ZYX')[0]

    def height_above_ground(self):
        pos = self.data.qpos[:3]
        ground = g1_model.ground_below(self.model, self.data, pos[0], pos[1], top=pos[2])
        return np.inf if ground is None else pos[2] - ground

    def control(self, t):
        command = np.zeros(3) if t < self.hold else self.motion.command(t - self.hold, *self.pelvis_pose())
        self.data.ctrl[:] = self.policy(self.data, command)

    def run(self):
        for step in range(int(SETTLE / IMU_DT)):
            if step % self.decimation == 0:
                self.control(0.0)
            mujoco.mj_step(self.model, self.data)
        torso = self.data.body('torso_link')
        rot = Rotation.from_matrix(torso.xmat.reshape(3, 3))
        roll, pitch = rot.as_euler('xyz', degrees=True)[:2]
        heading = rot.as_euler('ZYX', degrees=True)[0]
        x, y, z = torso.xpos
        self.get_logger().info(
            f'Standing after {SETTLE:.0f} s: pelvis {self.height_above_ground():.3f} m above the ground, torso roll '
            f'{roll:+.2f} pitch {pitch:+.2f} deg (gravity_align levels odom for this). odom will be about torso_link '
            f'now, levelled: ({x:.3f}, {y:.3f}, {z:.3f}) m in the scene, heading {heading:.2f} deg')

        self.get_logger().info('Waiting for a /livox/lidar subscriber (start perception.launch.py)...')
        while rclpy.ok() and self.count_subscribers('/livox/lidar') == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
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

            policy_tick = tick % self.decimation == 0
            if policy_tick:
                self.control(t)
            clock = Clock()
            clock.clock = to_stamp(t)
            self.clock_pub.publish(clock)
            if policy_tick:
                self.publish_joints(t)
            # After mj_step, positions and sensors are still those at t; only qpos and qvel have moved on.
            mujoco.mj_step(self.model, self.data)
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

            if policy_tick and self.height_above_ground() < FALLEN_HEIGHT:
                fell = True
                self.get_logger().error(f'The robot fell at t = {t:.1f} s, near ({self.data.qpos[0]:.2f}, '
                                        f'{self.data.qpos[1]:.2f}), after {self.motion.progress}.')
                break
            if finished is None and t >= self.hold and self.motion.done:
                finished = t
                self.get_logger().info(f'Motion finished at t = {t:.1f} s.')
            if finished is not None and t > finished + 2.0:
                break
            if viewer is not None and tick % 6 == 0:
                viewer.sync()
            rclpy.spin_once(self, timeout_sec=0.0)
            tick += 1

        end = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        realtime = tick * IMU_DT / max(time.monotonic() - wall_start - 2.0, 1e-3)
        if not fell:
            self.get_logger().info(f'Completed {self.motion.progress}. Ran at {realtime:.2f}x real time.')
        self.report()
        if isinstance(self.scan, ElevationMapScan):
            self.get_logger().info(self.scan.report())
        if viewer is not None:
            viewer.close()


def main():
    rclpy.init()
    node = G1WalkSim()
    try:
        node.run()
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
