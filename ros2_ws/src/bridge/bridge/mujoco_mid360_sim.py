"""G1 on a scripted path in MuJoCo, publishing a simulated Mid-360 in the real driver's format.

Publishes /clock, /livox/imu, /livox/lidar (CustomMsg), /joint_states and /ground_truth/odom, and on exit reports
FAST-LIO's /Odometry error against the true pose. The robot is posed kinematically, not simulated
with physics: there is no controller yet. The base follows a circle with a gait-like bob and sway.
"""
from pathlib import Path
import time

from livox_ros_driver2.msg import CustomMsg, CustomPoint
import mujoco
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, JointState

IMU_DT = 0.005      # 200 Hz, as on the Mid-360
JOINT_TICKS = 4     # publish joint states at 50 Hz
SLICE_TICKS = 2     # trace a slice of the scan every 10 ms, so points get real per-point times
FRAME_SLICES = 10   # 10 Hz frames
FD_STEP = 1e-3      # s, for differentiating the path
STAMP_OFFSET = 100.0  # s, ROS treats time 0 as unset
GRAVITY = 9.81


def default_scene():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'sim' / 'g1_lidar_scene.xml').exists():
            return str(parent / 'sim' / 'g1_lidar_scene.xml')
    return ''


def to_stamp(t):
    ns = int(round((t + STAMP_OFFSET) * 1e9))
    return rclpy.time.Time(nanoseconds=ns).to_msg()


class ScriptedPath:
    """Still for `hold` s so FAST-LIO can initialise, then ramps up to a circle starting along +x."""

    RADIUS = 2.5
    SPEED = 0.5
    RAMP = 3.0
    STEP_HZ = 1.8

    def __init__(self, hold, z0, yaw0=0.0):
        self.hold = hold
        self.z0 = z0

    def pose(self, t):
        tau = max(t - self.hold, 0.0)
        x = min(tau / self.RAMP, 1.0)
        rate = self.SPEED / self.RADIUS
        if tau < self.RAMP:
            theta = rate * self.RAMP * (x**3 - x**4 / 2)
        else:
            theta = rate * (self.RAMP / 2 + tau - self.RAMP)
        amp = 3 * x**2 - 2 * x**3
        step = 2 * np.pi * self.STEP_HZ * tau
        pos = np.array([
            self.RADIUS * np.sin(theta),
            self.RADIUS * (1 - np.cos(theta)),
            self.z0 + amp * 0.02 * np.sin(step),
        ])
        rot = Rotation.from_euler('ZYX', [theta, amp * 0.02 * np.sin(step), amp * 0.03 * np.sin(step / 2)])
        return pos, rot


class Mid360Sim(Node):

    def __init__(self):
        super().__init__('mujoco_mid360_sim')
        # Basic Setup
        self.scene = self.declare_parameter('scene', default_scene()).value #sim/g1_lidar_scene.xml
        self.viewer = self.declare_parameter('viewer', True).value
        self.duration = self.declare_parameter('duration', 60.0).value
        self.hold = self.declare_parameter('hold', 3.0).value
        self.rays_per_slice = self.declare_parameter('rays_per_slice', 2000).value
        seed = self.declare_parameter('seed', 0).value
        self.rng = np.random.default_rng(seed)
        self.declare_lidar_noise(seed)

        # MuJoCo setup
        self.model = mujoco.MjModel.from_xml_path(self.scene)
        self.data = mujoco.MjData(self.model)
        # Only kinematics and sensors are needed from mj_inverse, not contact forces.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

        # MuJoCo lidar https://github.com/discoverse-dev/MuJoCo-LiDAR
        # Imported here so a missing simulator dependency gives this clear error, not a build failure.
        from mujoco_lidar import MjLidarWrapper
        from mujoco_lidar.scan_gen import LivoxGenerator
        # The site is inside the head mesh, which belongs to torso_link, so tracing that body would
        # stop every ray about 3 cm out. The arms and legs stay visible, as on the real robot.
        self.lidar = MjLidarWrapper(
            self.model, 'mid360', backend='warp', cutoff_dist=40.0,
            args={'bodyexclude': self.model.body('torso_link').id})
        self.rays = LivoxGenerator('mid360').ray_angles.astype(np.float32)
        self.ray_index = 0

        # IMU sensor, joints, and scripted path   
        self.gyro = self.model.sensor('mid360-imu-angular-velocity')
        self.accel = self.model.sensor('mid360-imu-linear-acceleration')
        self.path = ScriptedPath(self.hold, self.model.qpos0[2])
        hinges = np.nonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.joint_names = [self.model.joint(int(j)).name for j in hinges]
        self.joint_qpos = self.model.jnt_qposadr[hinges]
        self.joint_dof = self.model.jnt_dofadr[hinges]

        # Publishers
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/livox/imu', 200)
        self.lidar_pub = self.create_publisher(CustomMsg, '/livox/lidar', 10)
        self.gt_pub = self.create_publisher(Odometry, '/ground_truth/odom', 10)
        self.create_subscription(Odometry, '/Odometry', self.on_odometry, 100)

        self.gt = []            # true IMU pose relative to the start, one per IMU tick
        self.t_start = None     # the same frame FAST-LIO's camera_init takes at startup
        self.errors = []

    def set_state(self, t):
        """
            Set the next state from pre-determined path, and calculate angles
        """
        h = FD_STEP
        pos, rot = self.path.pose(t) # get the path at time t
        pos_next, rot_next = self.path.pose(t + h)
        pos_prev, rot_prev = self.path.pose(t - h)
        omega = (rot_prev.inv() * rot_next).as_rotvec() / (2 * h)
        omega_next = (rot.inv() * self.path.pose(t + 2 * h)[1]).as_rotvec() / (2 * h)
        omega_prev = (self.path.pose(t - 2 * h)[1].inv() * rot).as_rotvec() / (2 * h)
        x, y, z, w = rot.as_quat()
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = [w, x, y, z]
        self.data.qvel[:3] = (pos_next - pos_prev) / (2 * h)
        self.data.qvel[3:6] = omega
        self.data.qacc[:3] = (pos_next - 2 * pos + pos_prev) / h**2
        self.data.qacc[3:6] = (omega_next - omega_prev) / (2 * h)
        # Compute inverse dynamics
        mujoco.mj_inverse(self.model, self.data)

    def sensor(self, sensor):
        value = self.data.sensordata[sensor.adr[0]:sensor.adr[0] + 3]
        return value + self.rng.normal(0.0, sensor.noise[0], 3)

    def publish_imu(self, t):
        msg = Imu()
        msg.header.stamp = to_stamp(t)
        msg.header.frame_id = 'livox_frame'
        gyro = self.sensor(self.gyro)
        accel = self.sensor(self.accel) / GRAVITY  # the Livox driver publishes acceleration in g
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, accel)
        self.imu_pub.publish(msg)

    def publish_joints(self, t):
        msg = JointState()
        msg.header.stamp = to_stamp(t)
        msg.name = self.joint_names
        msg.position = self.data.qpos[self.joint_qpos].tolist()
        msg.velocity = self.data.qvel[self.joint_dof].tolist()
        self.joint_pub.publish(msg)

    def record_ground_truth(self, t):
        """
            Record gt pose using the mid360_imu, and publish as an Odometry message
        """
        site = self.data.site('mid360_imu')
        pos = site.xpos.copy()
        rot = Rotation.from_matrix(site.xmat.reshape(3, 3).copy())
        if self.t_start is None:
            self.t_start = (pos, rot)
        rel_pos = self.t_start[1].inv().apply(pos - self.t_start[0])
        rel_rot = self.t_start[1].inv() * rot
        self.gt.append((rel_pos, rel_rot))

        msg = Odometry()
        msg.header.stamp = to_stamp(t)
        msg.header.frame_id = 'camera_init'
        msg.child_frame_id = 'mid360_imu_gt'
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, rel_pos)
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, rel_rot.as_quat())
        self.gt_pub.publish(msg)

    def declare_lidar_noise(self, seed):
        """LiDAR noise parameters, from the Mid-360's spec. noise:=false turns off all but the blind zone."""
        self.noise = self.declare_parameter('noise', True).value
        self.range_noise = self.declare_parameter('range_noise', 0.02).value     # m, 1 sigma, along the ray
        self.angle_noise = np.radians(self.declare_parameter('angle_noise', 0.15).value)  # deg, 1 sigma per axis
        self.dropout = self.declare_parameter('dropout', 0.02).value             # fraction of returns lost
        self.blind_zone = self.declare_parameter('blind_zone', 0.1).value        # m, closer returns are dropped
        # Its own generator, so turning LiDAR noise on or off leaves the IMU noise unchanged.
        self.lidar_rng = np.random.default_rng(seed + 1)

    def add_lidar_noise(self, dist, hits, keep):
        """Measured range = true range + Gaussian noise. The direction error is modelled by reporting the
        point slightly off the true ray, perpendicular to it, which is what an angular encoder error does."""
        n = len(dist)
        rng = self.lidar_rng
        # Draw for every ray, hit or not, so the noise doesn't depend on what the scan hits.
        angle_err = rng.normal(0.0, self.angle_noise, (n, 2))
        range_err = rng.normal(0.0, self.range_noise, n)
        keep = keep & (rng.random(n) >= self.dropout)
        hit = np.nonzero(keep)[0]   # misses have no direction to perturb
        dirs = hits[hit] / dist[hit, None]
        ref = np.where(np.abs(dirs[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        e1 = np.cross(dirs, ref)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(dirs, e1)
        dirs = dirs + angle_err[hit, :1] * e1 + angle_err[hit, 1:] * e2
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        dist, hits = dist.copy(), hits.copy()
        dist[hit] += range_err[hit]
        hits[hit] = dirs * dist[hit, None]
        return dist, hits, keep

    def trace_slice(self, offset_ns, line_start):
        end = self.ray_index + self.rays_per_slice
        rays = self.rays[np.arange(self.ray_index, end) % len(self.rays)]
        self.ray_index = end % len(self.rays)
        dist = np.asarray(self.lidar.trace_rays(self.data, rays[:, 0], rays[:, 1]), dtype=float)
        hits = np.asarray(self.lidar.get_hit_points(), dtype=float)
        keep = dist > 0.0   # misses are -1
        if self.noise:
            dist, hits, keep = self.add_lidar_noise(dist, hits, keep)
        points = []
        for i in np.nonzero(keep & (dist > self.blind_zone))[0]:
            x, y, z = hits[i]
            points.append(CustomPoint(
                offset_time=offset_ns + int(i) * int(1e7 / self.rays_per_slice),
                x=float(x), y=float(y), z=float(z),
                reflectivity=100, tag=0x10, line=int(line_start + i) % 4,
            ))
        return points

    def on_odometry(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - STAMP_OFFSET
        index = int(round(t / IMU_DT))
        if not 0 <= index < len(self.gt):
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        gt_pos, gt_rot = self.gt[index]
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.errors.append((t, np.linalg.norm(np.array([p.x, p.y, p.z]) - gt_pos),
                            np.degrees((gt_rot.inv() * rot).magnitude())))

    def report(self):
        if not self.errors:
            self.get_logger().error('No /Odometry received from FAST-LIO, so nothing to compare.')
            return
        e = np.array(self.errors)
        path = sum(np.linalg.norm(b[0] - a[0]) for a, b in zip(self.gt[::20], self.gt[20::20]))
        self.get_logger().info(
            f'FAST-LIO vs ground truth over {path:.1f} m, {len(e)} poses: '
            f'translation RMSE {np.sqrt(np.mean(e[:, 1]**2)):.3f} m, max {e[:, 1].max():.3f} m, '
            f'final {e[-1, 1]:.3f} m | rotation RMSE {np.sqrt(np.mean(e[:, 2]**2)):.2f} deg, '
            f'max {e[:, 2].max():.2f} deg, final {e[-1, 2]:.2f} deg')

    def run(self):
        self.get_logger().info('Waiting for a /livox/lidar subscriber (start perception.launch.py)...')
        while rclpy.ok() and self.count_subscribers('/livox/lidar') == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'Running for {self.duration:.0f} s of sim time.')

        viewer = None
        if self.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        wall_start = time.monotonic()
        points = []
        frame_start = 0.0
        slices = 0
        tick = 0
        while rclpy.ok() and tick * IMU_DT < self.duration and (viewer is None or viewer.is_running()):
            t = tick * IMU_DT
            ahead = wall_start + t - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)

            # Setup: set the state (pose in sim), then publish clock/imu, then publish gt pose
            self.set_state(t)
            clock = Clock()
            clock.clock = to_stamp(t)
            self.clock_pub.publish(clock)
            self.publish_imu(t)
            self.record_ground_truth(t)
            if tick % JOINT_TICKS == 0:
                self.publish_joints(t)
            # Publish LiDAR points here. First check every other tick
            if tick % SLICE_TICKS == 0:
                # Every 10 slices (10Hz), publish accumulated LiDAR points (10 slices)
                if slices == FRAME_SLICES:
                    msg = CustomMsg()
                    msg.header.stamp = to_stamp(frame_start)
                    msg.header.frame_id = 'livox_frame'
                    msg.timebase = int(round((frame_start + STAMP_OFFSET) * 1e9))
                    msg.point_num = len(points)
                    msg.points = points
                    self.lidar_pub.publish(msg)
                    points, slices, frame_start = [], 0, t #reset
                offset_ns = int(round((t - frame_start) * 1e9))
                # Accumulate LiDAR points here
                points += self.trace_slice(offset_ns, slices * self.rays_per_slice)
                slices += 1

            if viewer is not None and tick % 6 == 0:
                viewer.sync()
            rclpy.spin_once(self, timeout_sec=0.0)
            tick += 1

        # Give FAST-LIO time to finish its last frames.
        end = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        realtime = tick * IMU_DT / max(time.monotonic() - wall_start - 2.0, 1e-3)
        self.get_logger().info(f'Ran at {realtime:.2f}x real time.')
        self.report()
        if viewer is not None:
            viewer.close()


def main():
    rclpy.init()
    node = Mid360Sim()
    try:
        node.run()
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
