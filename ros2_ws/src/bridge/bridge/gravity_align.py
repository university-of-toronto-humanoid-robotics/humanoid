"""Publishes a gravity-aligned odom -> camera_init once the IMU has measured gravity.

camera_init is the Mid-360 IMU's pose when FAST-LIO starts. A fixed odom -> camera_init assumes the
torso is level at that moment, so any tilt then tilts the whole map: 1 degree is 1.7 cm per metre.
This node averages the accelerometer while the robot stands still at startup, which gives the up
direction in the IMU frame, and levels odom with it. odom keeps the torso's heading and origin.

The mount (torso_link -> mid360_imu, as x/y/z/roll/pitch/yaw) comes from parameters. Leave them at
zero for an upright LiDAR on its own, where odom is then the levelled IMU frame.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster


class GravityAlign(Node):

    def __init__(self):
        super().__init__('gravity_align')
        mount = [self.declare_parameter(name, 0.0).value for name in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        self.mount_t = np.array(mount[:3])
        self.mount_r = Rotation.from_euler('xyz', mount[3:])   # the same RPY convention as static_transform_publisher
        self.samples = self.declare_parameter('samples', 200).value   # 1 s of the Mid-360's 200 Hz IMU
        self.accel = []
        self.broadcaster = StaticTransformBroadcaster(self)
        self.sub = self.create_subscription(Imu, '/livox/imu', self.on_imu, 100)

    def on_imu(self, msg):
        a = msg.linear_acceleration
        self.accel.append((a.x, a.y, a.z))
        # Until we reach 200 samples, simply accumulate accelerometer readings
        if len(self.accel) < self.samples:
            return
        # Once we reach 200 samples, don't subscribe anymore, and publish the one-time correction
        self.destroy_subscription(self.sub)

        # Standing still, the accelerometer measures the reaction to gravity, so it points up.
        # Only its direction is used, so the Livox driver's units of g don't matter.
        # First, apply rotation of mount wrt odom to the average (x,y,z) accel readings from livox IMU. Accel reading wrt torso_link
        up = self.mount_r.apply(np.mean(self.accel, axis=0))
        roll = np.arctan2(up[1], up[2])
        pitch = np.arctan2(-up[0], np.hypot(up[1], up[2]))
        level = Rotation.from_euler('xyz', [roll, pitch, 0.0])   # the torso's attitude in odom, without yaw
        rotation = level * self.mount_r
        translation = level.apply(self.mount_t)

        # Broadcast the correct odom->camera_init transform, where odom is z-up, while torso may not be at startup.
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'camera_init'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, translation)
        q = t.transform.rotation
        q.x, q.y, q.z, q.w = map(float, rotation.as_quat())
        self.broadcaster.sendTransform(t)
        self.get_logger().info(f'odom levelled from {len(self.accel)} IMU samples: torso roll '
                               f'{np.degrees(roll):+.2f} deg, pitch {np.degrees(pitch):+.2f} deg')


def main():
    rclpy.init()
    try:
        rclpy.spin(GravityAlign())   # keeps running so late subscribers still get the static transform
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
