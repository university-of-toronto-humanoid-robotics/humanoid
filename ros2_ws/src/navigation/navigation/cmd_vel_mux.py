"""Lets one source drive at a time: forwards the operator's or the navigator's velocity command to /cmd_vel.

In:  /cmd_vel/teleop   the operator (keyboard_teleop, gamepad_teleop). It sends zeros even while locked.
     /cmd_vel/nav      the navigator
Out: /cmd_vel          for walk_controller

The operator always wins. A non-zero teleop command takes over at once and stops the navigator (/navigator/stop),
so an operator touching a stick is a takeover, not a tug of war. The operator keeps control, zeros included,
until teleop has sent only zeros for HOLD seconds. The navigator then stays stopped until someone starts it again.
Teleop's zeros while nobody touches it don't block the navigator.

Nothing is forwarded while neither source is sending, and walk_controller then stands still by itself.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger

HOLD = 1.0   # s of teleop zeros after which the operator lets go of control


def moving(msg):
    return any(v != 0.0 for v in (msg.linear.x, msg.linear.y, msg.linear.z,
                                  msg.angular.x, msg.angular.y, msg.angular.z))


class Mux:
    """Which source's message to forward. No ROS clock in here, so it can be tested."""

    def __init__(self, hold=HOLD):
        self.hold = hold
        self.operator_moved = None   # time of the operator's last non-zero command

    def operator_has_control(self, now):
        return self.operator_moved is not None and now - self.operator_moved < self.hold

    def teleop(self, msg, now):
        """(forward this message?, did the operator just take over?)"""
        took_over = False
        if moving(msg):
            took_over = not self.operator_has_control(now)
            self.operator_moved = now
        return self.operator_has_control(now), took_over

    def nav(self, msg, now):
        return not self.operator_has_control(now)


class CmdVelMux(Node):

    def __init__(self):
        super().__init__('cmd_vel_mux')
        self.mux = Mux(self.declare_parameter('hold', HOLD).value)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_navigator = self.create_client(Trigger, '/navigator/stop')
        self.create_subscription(Twist, '/cmd_vel/teleop', self.on_teleop, 10)
        self.create_subscription(Twist, '/cmd_vel/nav', self.on_nav, 10)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_teleop(self, msg):
        forward, took_over = self.mux.teleop(msg, self.now())
        if took_over:
            self.get_logger().warn('The operator took over: stopping the navigator.')
            if self.stop_navigator.service_is_ready():
                self.stop_navigator.call_async(Trigger.Request())
        if forward:
            self.pub.publish(msg)

    def on_nav(self, msg):
        if self.mux.nav(msg, self.now()):
            self.pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(CmdVelMux())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
