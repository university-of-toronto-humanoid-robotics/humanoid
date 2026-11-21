"""Robot status for the operator's status line, from the robot's own topics.

In:  /lowstate     the pelvis IMU (unitree_hg/LowState): attitude and turn rate
     TF odom -> pelvis   FAST-LIO's pose of the robot, for its walking speed
Out: /g1/tilt      std_msgs/Float32, degrees from upright
     /g1/velocity  geometry_msgs/TwistStamped: forward and sideways speed (m/s) in the robot's heading
                   frame, and turn rate (rad/s), the same axes as /cmd_vel

Runs on the robot side (bringup g1.launch.py), sim or real: it only uses what the real G1 and its
software provide. It publishes only while /lowstate is arriving, so a dead robot link shows up in
teleop as ROBOT LINK LOST rather than as a robot standing still.
"""
from __future__ import annotations

import math
from collections import deque

PUBLISH_HZ = 20.0
LOWSTATE_TIMEOUT = 0.5   # s without /lowstate before going quiet
SPEED_WINDOW = 0.3       # s of FAST-LIO poses the walking speed is averaged over


def tilt_deg(quat_wxyz) -> float:
    """Angle between the body's up axis and the world's, from a (w, x, y, z) attitude."""
    w, x, y, z = quat_wxyz
    norm = w * w + x * x + y * y + z * z
    if norm == 0.0:
        return 0.0
    up_z = 1.0 - 2.0 * (x * x + y * y) / norm   # world z of the body's z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def yaw_of(x: float, y: float, z: float, w: float) -> float:
    """Heading (rad) of an (x, y, z, w) rotation in a z-up frame."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SpeedEstimator:
    """Walking speed in the robot's heading frame, from timestamped planar poses."""

    def __init__(self, window: float = SPEED_WINDOW):
        self.window = window
        self.poses: deque[tuple[float, float, float, float]] = deque()

    def add(self, t: float, x: float, y: float, yaw: float) -> None:
        if self.poses and t <= self.poses[-1][0]:
            if t < self.poses[-1][0]:   # time went backwards (sim restarted): start over
                self.poses.clear()
            else:
                return                  # the same pose again
        self.poses.append((t, x, y, yaw))
        while len(self.poses) > 2 and t - self.poses[1][0] >= self.window:
            self.poses.popleft()

    def speed(self) -> tuple[float, float] | None:
        """(forward, sideways) in m/s, or None until two poses span some time."""
        if len(self.poses) < 2:
            return None
        t0, x0, y0, _ = self.poses[0]
        t1, x1, y1, yaw = self.poses[-1]
        if t1 - t0 <= 0.0:
            return None
        vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
        c, s = math.cos(yaw), math.sin(yaw)
        return c * vx + s * vy, -s * vx + c * vy


def main():
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from std_msgs.msg import Float32
    from tf2_ros import Buffer, TransformException, TransformListener
    from unitree_hg.msg import LowState

    rclpy.init()
    node = rclpy.create_node('robot_status')
    buffer = Buffer()
    TransformListener(buffer, node)
    tilt_pub = node.create_publisher(Float32, '/g1/tilt', 10)
    velocity_pub = node.create_publisher(TwistStamped, '/g1/velocity', 10)
    state = {'tilt': 0.0, 'turn': 0.0, 'seen': None}
    speed = SpeedEstimator()

    def now() -> float:
        return node.get_clock().now().nanoseconds * 1e-9

    def on_lowstate(msg):
        state['tilt'] = tilt_deg(msg.imu_state.quaternion)
        state['turn'] = float(msg.imu_state.gyroscope[2])
        state['seen'] = now()

    def tick():
        if state['seen'] is None or now() - state['seen'] > LOWSTATE_TIMEOUT:
            return
        try:
            tf = buffer.lookup_transform('odom', 'pelvis', Time())
            p, q = tf.transform.translation, tf.transform.rotation
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            speed.add(stamp, p.x, p.y, yaw_of(q.x, q.y, q.z, q.w))
        except TransformException:
            node.get_logger().warn('No odom -> pelvis yet (is FAST-LIO running?): speed shows 0.',
                                   throttle_duration_sec=10.0)
        vx, vy = speed.speed() or (0.0, 0.0)
        tilt_pub.publish(Float32(data=state['tilt']))
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = vx, vy, state['turn']
        velocity_pub.publish(msg)

    node.create_subscription(LowState, '/lowstate', on_lowstate, qos_profile_sensor_data)
    node.create_timer(1.0 / PUBLISH_HZ, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
