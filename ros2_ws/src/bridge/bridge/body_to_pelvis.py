"""Publishes body -> pelvis, joining FAST-LIO's TF tree to the URDF's.

FAST-LIO's body and the URDF's mid360_imu are the same physical point, but TF gives each frame one
parent, so they can't be merged. The pelvis's pose relative to mid360_imu is therefore also its pose
relative to body. It follows the waist joints, so it has to be updated from /joint_states.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


class BodyToPelvis(Node):

    def __init__(self):
        super().__init__('body_to_pelvis')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.last_stamp = None
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)

    def on_joint_states(self, _):
        try:
            transform = self.buffer.lookup_transform('mid360_imu', 'pelvis', Time())
        except TransformException:
            return  # robot_state_publisher hasn't published the waist joints yet
        # robot_state_publisher can publish less often than joint states arrive, and TF rejects repeats.
        stamp = (transform.header.stamp.sec, transform.header.stamp.nanosec)
        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp
        transform.header.frame_id = 'body'
        transform.child_frame_id = 'pelvis'
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    try:
        rclpy.spin(BodyToPelvis())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # ros2 launch can send a second SIGINT, so don't start any cleanup that it could interrupt


if __name__ == '__main__':
    main()
