"""G1 walking a route through the LiDAR room in MuJoCo under a trained policy, with a simulated Mid-360.

Like mujoco_mid360_sim, but with physics instead of a scripted pose: a flat-ground walking policy trained in
mjlab (task Unitree-G1-Flat, an rsl_rl checkpoint) drives all 29 joints while the robot follows waypoints around
the room, beside the stairs and boxes. The robot model, actuators, contacts and solver settings reproduce how
mjlab built the robot it was trained on (unitree_rl_mjlab's g1_constants.py and velocity task), so the policy
sees the same robot.

Publishes the same topics as mujoco_mid360_sim (/clock, /livox/imu, /livox/lidar, /joint_states and
/ground_truth/odom) and reports FAST-LIO's error at the end. Navigation uses the true pose.
"""
from pathlib import Path
import re
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

from bridge.mujoco_mid360_sim import FRAME_SLICES, IMU_DT, SLICE_TICKS, STAMP_OFFSET, Mid360Sim, to_stamp

DECIMATION = 4              # physics steps per policy step: 50 Hz policy, as in training
POLICY_DT = DECIMATION * IMU_DT
GAIT_PERIOD = 0.6           # s, the policy's phase clock
SETTLE = 1.0                # s of standing before sim time starts, so the drop onto the floor isn't recorded
FALLEN_HEIGHT = 0.45        # m, pelvis height below which the robot has fallen

# Actuators, from unitree_rl_mjlab's g1_constants.py: each joint is a <position> actuator whose gains come from
# its motor's reflected inertia (armature), a 10 Hz natural frequency and a damping ratio of 2.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def two_stage_planetary_armature(rotor, gears):
    return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


ARMATURE_5020 = two_stage_planetary_armature((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = two_stage_planetary_armature((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = two_stage_planetary_armature((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
ARMATURE_4010 = two_stage_planetary_armature((0.068e-4, 0.0, 0.0), (1, 5, 5))
MOTORS = (  # (joint names, armature, effort limit in N m). Waist pitch/roll and ankles are two 5020s each.
    (r'.*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint', ARMATURE_5020, 25.0),
    (r'.*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint', ARMATURE_7520_14, 88.0),
    (r'.*_hip_roll_joint|.*_knee_joint', ARMATURE_7520_22, 139.0),
    (r'.*_wrist_(pitch|yaw)_joint', ARMATURE_4010, 5.0),
    (r'waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint', 2 * ARMATURE_5020, 50.0),
)
# The standing pose the policy's actions are offsets from (HOME_KEYFRAME); other joints are 0.
DEFAULT_POSE = {
    r'.*_hip_pitch_joint': -0.1, r'.*_knee_joint': 0.3, r'.*_ankle_pitch_joint': -0.2,
    r'.*_shoulder_pitch_joint': 0.35, r'.*_elbow_joint': 0.87,
    r'left_shoulder_roll_joint': 0.18, r'right_shoulder_roll_joint': -0.18,
}
START_HEIGHT = 0.8
FOOT_GEOM = r'(left|right)_foot[1-7]_collision'

# Waypoints (x, y) in the room, starting and ending at the origin, at least 1 m from obstacles.
ROUTE = ((2.5, 0.0), (3.6, 2.5), (3.6, 5.5), (1.0, 7.5), (-3.0, 7.5), (-8.0, 7.0), (-8.5, 2.0),
         (-8.5, -2.0), (-6.0, -5.0), (-2.0, -5.0), (2.0, -4.8), (1.5, -1.0), (0.0, 0.0))


def find_in_repo(relative):
    for parent in Path(__file__).resolve().parents:
        if (parent / relative).exists():
            return str(parent / relative)
    return ''


def match_one(table, name):
    hits = [value for pattern, *value in table if re.fullmatch(pattern, name)]
    if len(hits) != 1:
        raise ValueError(f'{name} matches {len(hits)} entries, expected 1')
    return hits[0]


def build_model(scene, description):
    """The scene's G1 set up the way mjlab sets it up for training, plus the Mid-360 from our description."""
    spec = mujoco.MjSpec.from_file(scene)
    robot = spec.body('pelvis')

    # Position actuators without control limits, force-limited to the motor's effort limit.
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        armature, effort = match_one(MOTORS, joint.name)
        stiffness = armature * NATURAL_FREQ**2
        damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        actuator = spec.add_actuator(name=joint.name, target=joint.name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = stiffness
        actuator.biasprm[1] = -stiffness
        actuator.biasprm[2] = -damping
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[:] = [-effort, effort]
        joint.armature = armature
        joint.frictionloss = 0.0

    # Collisions as in mjlab's FULL_COLLISION: only the *_collision capsules, self-collisions included.
    for geom in robot.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if re.match(r'.*_collision', geom.name):
            foot = re.match(FOOT_GEOM, geom.name) is not None
            geom.contype, geom.conaffinity = 1, 1
            geom.condim = 3 if foot else 1
            geom.priority = 1 if foot else 0
            if foot:
                geom.friction[0] = 0.6
        else:
            geom.contype, geom.conaffinity = 0, 0

    # The Mid-360 and its IMU, copied from our description so the two sims can't drift apart.
    mount = mujoco.MjSpec.from_file(description)
    torso = spec.body('torso_link')
    for name in ('mid360', 'mid360_imu'):
        site = mount.site(name)
        torso.add_site(name=name, pos=site.pos, quat=site.quat, size=site.size)
    for name in ('mid360-imu-angular-velocity', 'mid360-imu-linear-acceleration'):
        sensor = mount.sensor(name)
        spec.add_sensor(name=name, type=sensor.type, objtype=sensor.objtype, objname=sensor.objname,
                        noise=sensor.noise)

    model = spec.compile()
    hinges = np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    assert (model.actuator_trnid[:, 0] == hinges).all(), 'actuators must follow joint order, like the policy'

    # Unitree-G1-Flat's solver settings, with mjlab's defaults for the rest.
    model.opt.timestep = IMU_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50
    model.opt.impratio = 1.0
    return model


class WalkPolicy:
    """An rsl_rl actor: its observation normaliser, then an ELU MLP. Returns the mean action, without sampling."""

    def __init__(self, path, model):
        import torch  # only to read the checkpoint
        actor = torch.load(path, map_location='cpu', weights_only=True)['actor_state_dict']
        self.mean = actor['obs_normalizer._mean'].numpy()[0]
        self.std = actor['obs_normalizer._std'].numpy()[0]
        ids = sorted({int(k.split('.')[1]) for k in actor if k.startswith('mlp.')})
        self.layers = [(actor[f'mlp.{i}.weight'].numpy(), actor[f'mlp.{i}.bias'].numpy()) for i in ids]

        hinges = np.nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        names = [model.joint(int(j)).name for j in hinges]
        self.default = np.array([next((v for p, v in DEFAULT_POSE.items() if re.fullmatch(p, n)), 0.0)
                                 for n in names])
        stiffness = np.array([match_one(MOTORS, n)[0] for n in names]) * NATURAL_FREQ**2
        effort = np.array([match_one(MOTORS, n)[1] for n in names])
        self.scale = 0.25 * effort / stiffness  # G1_ACTION_SCALE
        self.last_action = np.zeros(len(names))
        self.steps = 0
        expected = 3 + 3 + 3 + 2 + 3 * len(names)
        if self.layers[0][0].shape[1] != expected or self.layers[-1][0].shape[0] != len(names):
            raise ValueError(f'{path} takes {self.layers[0][0].shape[1]} observations and gives '
                             f'{self.layers[-1][0].shape[0]} actions; this sim provides {expected} and {len(names)}')

    def observation(self, data, command):
        """Unitree-G1-Flat's actor observations, in order."""
        w, x, y, z = data.qpos[3:7]
        gravity = Rotation.from_quat([x, y, z, w]).inv().apply([0.0, 0.0, -1.0])
        phase = (self.steps * POLICY_DT) % GAIT_PERIOD / GAIT_PERIOD
        clock = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)] if np.linalg.norm(command) >= 0.1 else [0, 0]
        return np.concatenate([
            data.qvel[3:6],  # pelvis angular velocity in the pelvis frame, which the pelvis gyro measures
            gravity, command, clock,
            data.qpos[7:] - self.default, data.qvel[6:], self.last_action,
        ])

    def __call__(self, data, command):
        """Joint position targets for the next policy step."""
        x = (self.observation(data, command) - self.mean) / (self.std + 1e-2)  # rsl_rl's normaliser, eps 1e-2
        for i, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))
        self.last_action = x
        self.steps += 1
        return self.default + self.scale * x


class RouteFollower:
    """Turns toward the next waypoint and walks when roughly facing it, within the policy's command ranges."""

    REACHED = 0.4   # m
    MAX_TURN = 0.5  # rad/s, the flat policy's play range

    def __init__(self, route, speed):
        self.route = route
        self.speed = speed
        self.index = 0

    def command(self, pos, yaw):
        while self.index < len(self.route):
            dx, dy = np.array(self.route[self.index]) - pos[:2]
            if np.hypot(dx, dy) > self.REACHED:
                break
            self.index += 1
        if self.index == len(self.route):
            return np.zeros(3)
        error = np.arctan2(np.sin(np.arctan2(dy, dx) - yaw), np.cos(np.arctan2(dy, dx) - yaw))
        forward = self.speed * max(np.cos(error), 0.0) ** 2 if abs(error) < 0.8 else 0.0
        return np.array([forward, 0.0, float(np.clip(1.0 * error, -self.MAX_TURN, self.MAX_TURN))])


class G1WalkSim(Node):

    # The sensor publishing and error report are the scripted sim's. They use self.data, rng, gyro, accel,
    # lidar, rays, ray_index, rays_per_slice, joint_*, gt, t_start, errors and the *_pub publishers.
    sensor = Mid360Sim.sensor
    publish_imu = Mid360Sim.publish_imu
    publish_joints = Mid360Sim.publish_joints
    record_ground_truth = Mid360Sim.record_ground_truth
    trace_slice = Mid360Sim.trace_slice
    on_odometry = Mid360Sim.on_odometry
    report = Mid360Sim.report

    def __init__(self):
        super().__init__('mujoco_g1_walk_sim')
        self.scene = self.declare_parameter('scene', find_in_repo('sim/g1_walk_scene.xml')).value
        description = self.declare_parameter(
            'description', find_in_repo('description/g1/g1_29dof_rev_1_0.xml')).value
        policy = self.declare_parameter('policy', find_in_repo('training/models/g1_walk/model_12400.pt')).value
        self.viewer = self.declare_parameter('viewer', True).value
        self.duration = self.declare_parameter('duration', 180.0).value
        self.hold = self.declare_parameter('hold', 3.0).value
        speed = self.declare_parameter('speed', 0.5).value
        self.rays_per_slice = self.declare_parameter('rays_per_slice', 2000).value
        self.rng = np.random.default_rng(self.declare_parameter('seed', 0).value)
        if not Path(policy).is_file():
            raise FileNotFoundError(
                f'No walking policy at "{policy}". Build with --symlink-install so the default '
                'training/models/g1_walk/model_12400.pt is found, or pass -p policy:=<path to a .pt checkpoint>')

        self.model = build_model(self.scene, description)
        self.data = mujoco.MjData(self.model)
        self.policy = WalkPolicy(policy, self.model)
        self.route = RouteFollower(ROUTE, speed)
        self.data.qpos[:3] = [0.0, 0.0, START_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:] = self.policy.default
        mujoco.mj_forward(self.model, self.data)

        from mujoco_lidar import MjLidarWrapper
        from mujoco_lidar.scan_gen import LivoxGenerator
        # Trace the room and the robot's visual meshes (groups 0-2), not its collision capsules (group 3).
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

    def control(self, t):
        command = np.zeros(3) if t < self.hold else self.route.command(*self.pelvis_pose())
        self.data.ctrl[:] = self.policy(self.data, command)

    def run(self):
        for _ in range(int(SETTLE / IMU_DT)):
            if _ % DECIMATION == 0:
                self.control(0.0)
            mujoco.mj_step(self.model, self.data)
        torso = Rotation.from_matrix(self.data.body('torso_link').xmat.reshape(3, 3)).as_euler('xyz', degrees=True)
        self.get_logger().info(
            f'Standing after {SETTLE:.0f} s: pelvis height {self.data.qpos[2]:.3f} m, torso roll {torso[0]:+.2f} '
            f'pitch {torso[1]:+.2f} deg (gravity_align levels odom for this)')

        self.get_logger().info('Waiting for a /livox/lidar subscriber (start perception.launch.py)...')
        while rclpy.ok() and self.count_subscribers('/livox/lidar') == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'Walking {len(ROUTE)} waypoints, for up to {self.duration:.0f} s of sim time.')

        viewer = None
        if self.viewer:
            from mujoco import viewer as mujoco_viewer
            viewer = mujoco_viewer.launch_passive(self.model, self.data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.model.body('pelvis').id
            viewer.cam.distance = 4.0

        wall_start = time.monotonic()
        points, frame_start, slices, tick = [], 0.0, 0, 0
        fell = False
        while rclpy.ok() and tick * IMU_DT < self.duration and (viewer is None or viewer.is_running()):
            t = tick * IMU_DT
            ahead = wall_start + t - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)

            if tick % DECIMATION == 0:
                self.control(t)
            clock = Clock()
            clock.clock = to_stamp(t)
            self.clock_pub.publish(clock)
            if tick % DECIMATION == 0:
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

            if self.data.qpos[2] < FALLEN_HEIGHT:
                fell = True
                self.get_logger().error(f'The robot fell at t = {t:.1f} s, near ({self.data.qpos[0]:.1f}, '
                                        f'{self.data.qpos[1]:.1f}), heading for waypoint {self.route.index + 1}.')
                break
            if self.route.index == len(ROUTE) and not hasattr(self, 'finished'):
                self.finished = t
                self.get_logger().info(f'Route finished at t = {t:.1f} s.')
            if hasattr(self, 'finished') and t > self.finished + 2.0:
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
            self.get_logger().info(f'Reached {self.route.index} of {len(ROUTE)} waypoints. Ran at {realtime:.2f}x real time.')
        self.report()
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
