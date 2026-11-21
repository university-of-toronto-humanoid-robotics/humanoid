"""Finds the robot on the competition course by lining the LiDAR scan up with the course map, and publishes map -> odom.

FAST-LIO says where the robot is relative to where it was switched on (odom). This node says where that start was
on the course (map), so map -> odom -> ... -> pelvis gives the robot's pose on the course. The course frame is the
one in sim/competition/README.md, which the sim uses as its world.

In:  /cloud_self_filtered  FAST-LIO's latest scan in its body frame, without the robot's own body (self_filter)
     TF odom -> body       FAST-LIO's pose, levelled by gravity_align
Out: TF map -> odom        20 times a second, once it has found itself
     ~/pose                the pelvis on the course (geometry_msgs/PoseStamped, frame map), after each match
     ~/map                 the map it matches against, for RViz (latched)

How:
  1. It keeps the last `window` seconds of scans, moved into odom with FAST-LIO's pose: one scan alone is sparse.
  2. Start: initial_pose says roughly where the pelvis is ([x, y, yaw] on the course). It tries starts within
     search_radius and search_yaw of that and keeps the one that fits best (icp.CourseMatcher.search).
  3. Then `rate` times a second it refines map -> odom from where it was. This is what removes FAST-LIO's drift.
  A match that fits less than min_fitness of the points, or moves the answer more than max_jump, is ignored and
  the last good map -> odom stays: FAST-LIO carries the robot until the next good match.
"""
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from localization.course_map import build_map, ground_offset, load_grid
from localization.icp import CourseMatcher, Pose

SCAN_VOXEL = 0.10      # m: the scan is thinned to one point per cube this size before matching
MAX_SCAN_POINTS = 4000


def to_matrix(transform):
    t, q = transform.translation, transform.rotation
    return np.array([t.x, t.y, t.z]), Rotation.from_quat([q.x, q.y, q.z, q.w])


def flat(t, rot):
    """A 3D pose cut down to x, y and heading."""
    return Pose(t[0], t[1], 0.0, rot.as_euler('ZYX')[0])


class Localizer(Node):

    def __init__(self):
        super().__init__('localizer')
        map_file = self.declare_parameter('map_file', '').value
        initial = list(self.declare_parameter('initial_pose', [0.0, 0.0, 0.0]).value)
        self.search_radius = self.declare_parameter('search_radius', 0.5).value
        self.search_yaw = np.radians(self.declare_parameter('search_yaw', 20.0).value)
        self.window = self.declare_parameter('window', 1.0).value
        self.min_fitness = self.declare_parameter('min_fitness', 0.5).value
        self.max_jump = self.declare_parameter('max_jump', 0.5).value
        self.max_yaw_jump = np.radians(self.declare_parameter('max_yaw_jump', 15.0).value)
        rate = self.declare_parameter('rate', 2.0).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.robot_frame = self.declare_parameter('robot_frame', 'pelvis').value
        if not map_file:
            raise RuntimeError('Set map_file to the course height grid (sim/competition/course_heightmap.npz)')

        self.grid = load_grid(map_file)
        points, normals = build_map(*self.grid)
        self.matcher = CourseMatcher(points, normals)
        self.initial = Pose(initial[0], initial[1], 0.0, np.radians(initial[2]) if len(initial) > 2 else 0.0)
        self.map_to_odom = None
        self.scans = deque()
        self.rejected = 0

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '~/pose', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(PointCloud2, '~/map', latched)
        self.map_pub.publish(point_cloud2.create_cloud_xyz32(
            Header(frame_id=self.map_frame), points.astype(np.float32)))
        self.create_subscription(PointCloud2, '/cloud_self_filtered', self.on_cloud, 10)
        self.create_timer(1.0 / rate, self.match)
        self.create_timer(0.05, self.publish_tf)
        self.get_logger().info(f'Course map: {len(points)} points. Waiting for scans; the pelvis should be near '
                               f'({self.initial.x:.2f}, {self.initial.y:.2f}) facing {initial[2]:.0f} deg.')

    def on_cloud(self, msg):
        try:
            t, rot = to_matrix(self.buffer.lookup_transform(self.odom_frame, msg.header.frame_id,
                                                            Time.from_msg(msg.header.stamp)).transform)
        except TransformException:
            return   # gravity_align's odom -> camera_init isn't up yet
        xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z')).astype(float)
        if len(xyz) == 0:
            return
        stamp = Time.from_msg(msg.header.stamp)
        self.scans.append((stamp, rot.apply(xyz) + t))
        while self.scans and (stamp - self.scans[0][0]).nanoseconds * 1e-9 > self.window:
            self.scans.popleft()

    def local_cloud(self):
        cloud = np.concatenate([points for _, points in self.scans])
        keys = np.floor(cloud / SCAN_VOXEL).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        cloud = cloud[first]
        if len(cloud) > MAX_SCAN_POINTS:
            cloud = cloud[np.random.default_rng().choice(len(cloud), MAX_SCAN_POINTS, replace=False)]
        return cloud

    def match(self):
        if not self.scans:
            return
        stamp = self.scans[-1][0]
        try:
            robot_t, robot_rot = to_matrix(
                self.buffer.lookup_transform(self.odom_frame, self.robot_frame, stamp).transform)
        except TransformException as e:
            self.get_logger().warn(f'No {self.odom_frame} -> {self.robot_frame}: {e}', throttle_duration_sec=5.0)
            return
        cloud = self.local_cloud()

        if self.map_to_odom is None:
            # The guess for map -> odom that puts the pelvis at initial_pose; its height from the floor.
            guess = self.initial.compose(flat(robot_t, robot_rot).inverse())
            guess.z = ground_offset(*self.grid, guess.apply(cloud))
            result = self.matcher.search(cloud, guess, self.search_radius, self.search_yaw)
            if result.fitness < self.min_fitness:
                self.get_logger().warn(f'Not found yet: the best start fits {result.fitness:.0%} of the scan (needs '
                                       f'{self.min_fitness:.0%}). Is initial_pose right?', throttle_duration_sec=5.0)
                return
            self.map_to_odom = result.pose
            self.get_logger().info(f'Found on the course: fits {result.fitness:.0%} of the scan, {result.rmse * 100:.1f} '
                                   f'cm RMS. {self.describe(robot_t, robot_rot)}')
        else:
            result = self.matcher.align(cloud, self.map_to_odom)
            moved, turned = result.pose.distance(self.map_to_odom)
            if result.fitness < self.min_fitness or moved > self.max_jump or turned > self.max_yaw_jump:
                self.rejected += 1
                self.get_logger().warn(
                    f'Match ignored ({self.rejected} so far): fits {result.fitness:.0%}, would move {moved:.2f} m '
                    f'and {np.degrees(turned):.1f} deg. Keeping the last one.', throttle_duration_sec=2.0)
                return
            self.map_to_odom = result.pose
        self.publish_pose(stamp, robot_t, robot_rot)

    def pelvis_on_course(self, robot_t, robot_rot):
        m = self.map_to_odom
        turn = Rotation.from_euler('z', m.yaw)
        return turn.apply(robot_t) + [m.x, m.y, m.z], turn * robot_rot

    def describe(self, robot_t, robot_rot):
        t, rot = self.pelvis_on_course(robot_t, robot_rot)
        return f'Pelvis at ({t[0]:.2f}, {t[1]:.2f}) facing {np.degrees(rot.as_euler("ZYX")[0]):.0f} deg.'

    def publish_pose(self, stamp, robot_t, robot_rot):
        t, rot = self.pelvis_on_course(robot_t, robot_rot)
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp.to_msg(), self.map_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, t)
        q = msg.pose.orientation
        q.x, q.y, q.z, q.w = map(float, rot.as_quat())
        self.pose_pub.publish(msg)

    def publish_tf(self):
        if self.map_to_odom is None:
            return
        m = self.map_to_odom
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = self.map_frame, self.odom_frame
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = m.x, m.y, m.z
        q = msg.transform.rotation
        q.x, q.y, q.z, q.w = map(float, Rotation.from_euler('z', m.yaw).as_quat())
        self.broadcaster.sendTransform(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(Localizer())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
