"""/lowstate (the G1's own joint report) to /joint_states, which robot_state_publisher turns into TF.

The same on the robot and in the sim. The G1 reports 500 times a second; TF doesn't need that, so this passes on
rate_hz of them.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from unitree_hg.msg import LowState

from policy.g1 import JOINTS


class JointStateBridge(Node):

    def __init__(self):
        super().__init__('joint_state_bridge')
        self.period = 1.0 / self.declare_parameter('rate_hz', 50.0).value
        self.last = None
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(LowState, '/lowstate', self.on_lowstate, qos_profile_sensor_data)

    def on_lowstate(self, low):
        now = self.get_clock().now()
        # 1 ms of slack, so jitter in an evenly spaced stream doesn't drop an extra message.
        if self.last is not None and (now - self.last).nanoseconds * 1e-9 < self.period - 0.001:
            return
        self.last = now
        msg = JointState()
        msg.header.stamp = now.to_msg()   # /lowstate carries no time of its own
        msg.name = JOINTS
        msg.position = [float(m.q) for m in low.motor_state[:len(JOINTS)]]
        msg.velocity = [float(m.dq) for m in low.motor_state[:len(JOINTS)]]
        msg.effort = [float(m.tau_est) for m in low.motor_state[:len(JOINTS)]]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
