"""Walks the G1 under a trained policy. The same node drives the real robot and the sim.

In:  /lowstate     the robot's joints and pelvis IMU (unitree_hg/LowState)
     /height_scan  the terrain scan, for policies that read the terrain (see the height_scan package)
     /cmd_vel      where to walk: forward and sideways speed (m/s) and turn rate (rad/s), in the robot's frame
Out: /lowcmd       a position target and gains for each motor, 50 times a second (unitree_hg/LowCmd)
     ~/state       wait, stand, walk or damp, whenever it changes

It moves through these states:
  wait   sends nothing, so the robot stays as it is: perception needs it still while it starts up. Leaves on a
         call to ~/start.
  stand  eases the joints from where they are to the standing pose over stand_time, then holds them there. These
         gains can't balance the robot, so it must be on the gantry.
  walk   the policy is in charge. Entered from stand by itself; a policy that reads the terrain first waits for
         a scan.
  damp   motors only resist motion, so the robot sinks down. Entered on a fault (the robot tipped over, or
         /lowstate stopped) or a call to ~/stop. It stays here until the node is restarted.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from policy import g1, lowcmd
from policy.walk_policy import gravity_in_body, make_policy

N = len(g1.JOINTS)
LOWSTATE_TIMEOUT = 0.1   # s without /lowstate before giving up
COMMAND_TIMEOUT = 0.5    # s without /cmd_vel before standing still
SCAN_TIMEOUT = 1.0       # s without /height_scan before standing still
MAX_TILT = 1.0           # rad from upright before giving up: the robot is falling
DAMP_KD = 3.0            # N m s/rad, the damping Unitree's own controller uses when passive
MOTOR_ON = 1


class WalkController(Node):

    def __init__(self):
        super().__init__('walk_controller')
        kind = self.declare_parameter('policy_type', 'rsl_rl_g1_flat').value
        checkpoint = self.declare_parameter('checkpoint', '').value
        scan_topic = self.declare_parameter('scan_topic', '/height_scan').value
        self.stand_time = self.declare_parameter('stand_time', 2.0).value
        self.started = False

        self.policy = make_policy(kind, checkpoint)
        self.default = g1.default_pose()
        self.kp, self.kd = g1.stiffness(), g1.damping()

        self.state = 'wait'
        self.low = self.low_time = None
        self.scan = self.scan_time = None
        self.command, self.command_time = np.zeros(3), None
        self.stand_from = self.stand_start = None

        # Sensor QoS so it hears the robot whichever way it publishes.
        self.create_subscription(LowState, '/lowstate', self.on_lowstate, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)
        if self.policy.needs_scan:
            self.create_subscription(Float32MultiArray, scan_topic, self.on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '~/state', latched)
        self.create_service(Trigger, '~/start', self.on_start)
        self.create_service(Trigger, '~/stop', self.on_stop)
        self.create_timer(self.policy.dt, self.step)
        self.state_pub.publish(String(data=self.state))
        self.get_logger().info(f'{kind} from {checkpoint}. Waiting for a call to ~/start.')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fresh(self, stamp, timeout):
        return stamp is not None and self.now() - stamp <= timeout

    def on_lowstate(self, msg):
        self.low, self.low_time = msg, self.now()

    def on_scan(self, msg):
        if len(msg.data) != self.policy.extra_obs:
            self.get_logger().error(f'The scan has {len(msg.data)} values, not {self.policy.extra_obs}: ignored.',
                                    throttle_duration_sec=5.0)
            return
        self.scan, self.scan_time = np.array(msg.data, dtype=float), self.now()

    def on_command(self, msg):
        command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        low, high = np.array(self.policy.COMMAND_RANGE).T
        self.command, self.command_time = np.clip(command, low, high), self.now()

    def on_start(self, request, response):
        response.success = self.state == 'wait' and self.low is not None
        if response.success:
            self.started = True
        else:
            response.message = 'no /lowstate yet' if self.state == 'wait' else f'already in state {self.state}'
        return response

    def on_stop(self, request, response):
        self.set_state('damp', 'stop requested')
        response.success = True
        return response

    def set_state(self, state, why=''):
        if state == self.state:
            return
        self.state = state
        self.state_pub.publish(String(data=state))
        log = self.get_logger().error if state == 'damp' else self.get_logger().info
        log(f'State: {state}' + (f' ({why})' if why else ''))

    def step(self):
        if self.low is None or (self.state == 'wait' and not self.started):
            return
        low = self.low
        joint_pos = np.array([m.q for m in low.motor_state[:N]])
        joint_vel = np.array([m.dq for m in low.motor_state[:N]])
        quat = np.array(low.imu_state.quaternion, dtype=float)
        gyro = np.array(low.imu_state.gyroscope, dtype=float)

        if self.state == 'wait':
            self.stand_from, self.stand_start = joint_pos, self.now()
            self.set_state('stand')
        if self.state != 'damp':
            if not self.fresh(self.low_time, LOWSTATE_TIMEOUT):
                self.set_state('damp', f'no /lowstate for {LOWSTATE_TIMEOUT} s')
            elif gravity_in_body(quat)[2] > -np.cos(MAX_TILT):
                self.set_state('damp', 'the robot tipped over')

        if self.state == 'stand':
            share = min((self.now() - self.stand_start) / max(self.stand_time, 1e-3), 1.0)
            target = self.stand_from + share * (self.default - self.stand_from)
            if share >= 1.0:
                if self.policy.needs_scan and not self.fresh(self.scan_time, SCAN_TIMEOUT):
                    self.get_logger().info('Standing until a terrain scan arrives.', throttle_duration_sec=5.0)
                else:
                    self.policy.reset()
                    self.set_state('walk')
        if self.state == 'walk':
            command = self.command if self.fresh(self.command_time, COMMAND_TIMEOUT) else np.zeros(3)
            if self.policy.needs_scan and not self.fresh(self.scan_time, SCAN_TIMEOUT):
                self.get_logger().warn('The terrain scan stopped: standing still on the last one.',
                                       throttle_duration_sec=2.0)
                command = np.zeros(3)
            target = self.policy(gyro, quat, joint_pos, joint_vel, command, self.scan)

        msg = LowCmd()
        msg.mode_pr, msg.mode_machine = low.mode_pr, low.mode_machine   # the robot rejects a mismatch
        for i in range(N):
            motor = msg.motor_cmd[i]
            motor.mode = MOTOR_ON
            if self.state == 'damp':
                motor.kd = DAMP_KD
            else:
                motor.q, motor.kp, motor.kd = float(target[i]), float(self.kp[i]), float(self.kd[i])
        msg.crc = lowcmd.crc(msg)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = WalkController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
