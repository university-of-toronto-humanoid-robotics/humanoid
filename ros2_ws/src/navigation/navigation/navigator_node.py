"""Walks the G1 along a route on the competition course: reads where the robot is, publishes /cmd_vel.

In:  TF map -> pelvis   where the robot is on the course (the localizer's map -> odom, then FAST-LIO and the URDF)
     /goal_pose         a single place to walk to and stop at (RViz's "2D Goal Pose" button publishes this)
Out: /cmd_vel           20 times a second while it is driving, for walk_controller
     ~/state            idle, following, aligning, waiting, arrived, stopped or lost (latched)
     ~/path             the route, for RViz (latched)
Services (std_srvs/Trigger):
     ~/start            walk the route in route_file from its first waypoint
     ~/stop             stop where it is. Zero /cmd_vel is sent, so the robot stands; it doesn't go limp

It does nothing until ~/start or a /goal_pose. If the pose is more than POSE_TIMEOUT old (the localizer or FAST-LIO
stopped), it sends zero /cmd_vel and reports lost until the pose comes back. How it steers: see follower.py.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from navigation.follower import Follower
from navigation.route import Route, Waypoint, load_route

RATE = 20.0          # Hz
POSE_TIMEOUT = 0.5   # s


class Navigator(Node):

    def __init__(self):
        super().__init__('navigator')
        route_file = self.declare_parameter('route_file', '').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.robot_frame = self.declare_parameter('robot_frame', 'pelvis').value
        self.goal_speed = self.declare_parameter('goal_speed', 0.3).value
        self.route = load_route(route_file) if route_file else None
        self.follower = None
        self.state = None

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '~/state', latched)
        self.path_pub = self.create_publisher(Path, '~/path', latched)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_service(Trigger, '~/start', self.on_start)
        self.create_service(Trigger, '~/stop', self.on_stop)
        self.create_timer(1.0 / RATE, self.step)
        self.set_state('idle')
        if self.route:
            self.publish_path(self.route)
            self.get_logger().info(f'Route "{self.route.name}": {len(self.route.waypoints)} waypoints. '
                                   'Call ~/start to walk it.')
        else:
            self.get_logger().info('No route_file: waiting for a /goal_pose.')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose(self):
        """(xy, yaw) of the robot on the course, or None if there is no fresh pose."""
        try:
            tf = self.buffer.lookup_transform(self.map_frame, self.robot_frame, Time())
        except TransformException:
            return None
        if self.now() - Time.from_msg(tf.header.stamp).nanoseconds * 1e-9 > POSE_TIMEOUT:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y]), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX')[0]

    def set_state(self, state, why=''):
        if state == self.state:
            return
        self.state = state
        self.state_pub.publish(String(data=state))
        log = self.get_logger().warn if state in ('lost', 'stopped') else self.get_logger().info
        log(f'State: {state}' + (f' ({why})' if why else ''))

    def begin(self, route):
        pose = self.pose()
        if pose is None:
            return f'no fresh {self.map_frame} -> {self.robot_frame}: is the localizer running and has it found the robot?'
        self.follower = Follower(route, pose[0])
        self.publish_path(route, pose[0])
        self.set_state('following', f'route "{route.name}", {len(route.waypoints)} waypoints')
        return ''

    def on_start(self, request, response):
        if self.route is None:
            response.success, response.message = False, 'no route_file was given'
            return response
        response.message = self.begin(self.route)
        response.success = not response.message
        return response

    def on_stop(self, request, response):
        self.follower = None
        self.cmd_pub.publish(Twist())
        self.set_state('stopped', 'stop requested')
        response.success = True
        return response

    def on_goal(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().error(f'Goal in frame "{msg.header.frame_id}"; it must be in {self.map_frame}.')
            return
        q = msg.pose.orientation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX')[0]
        goal = Waypoint(msg.pose.position.x, msg.pose.position.y, yaw=yaw, stop=True)
        why = self.begin(Route([goal], self.goal_speed, 'goal'))
        if why:
            self.get_logger().error(f'Goal ignored: {why}')

    def step(self):
        if self.follower is None:
            return
        pose = self.pose()
        if pose is None:
            self.cmd_pub.publish(Twist())
            self.set_state('lost', f'no {self.map_frame} -> {self.robot_frame} for {POSE_TIMEOUT} s: standing still')
            return
        command = self.follower.update(self.now(), *pose)
        if self.follower.state in ('waiting', 'arrived') and self.state == 'aligning' and self.follower.loose:
            self.get_logger().warn('Could not line up within the tolerance in time: accepted twice the tolerance.')
        self.set_state(self.follower.state, self.follower.progress)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = map(float, command)
        self.cmd_pub.publish(msg)
        if self.follower.done:
            target = self.follower.target
            error = np.linalg.norm(pose[0] - target.xy)
            self.get_logger().info(f'Arrived at ({target.x:.2f}, {target.y:.2f}): {error * 100:.1f} cm off, by its '
                                   'own estimate.')
            self.follower = None

    def publish_path(self, route, start=None):
        msg = Path()
        msg.header.frame_id = self.map_frame
        points = ([start] if start is not None else []) + [w.xy for w in route.waypoints]
        for xy in points:
            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x, pose.pose.position.y = map(float, xy)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(Navigator())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
