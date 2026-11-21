"""Keyboard teleoperation for the G1, run at the operator station.

The operator may not look at the robot (competition rules), so this terminal is
also their status display: what they command, what the robot actually does, how
far it is tilted, and whether the link to the robot is alive. The camera feed is
the other window (launch/operator.launch.py).

  ros2 run teleop keyboard_teleop

Keys (each press nudges the speed one step; terminals can't tell when a key is
released, so holding keys isn't used):
  w / s   forward / back        a / d   left / right        q / e   turn left / right
  space   stop                  Ctrl+C  stop and quit

Publishes:
  /cmd_vel/teleop      geometry_msgs/Twist   20 Hz, clamped to what the policy was trained on. bringup's
                       cmd_vel_mux passes it on to /cmd_vel: a non-zero command takes over from the navigator
  /g1/autonomy_mode    std_msgs/String       "teleop" (the declared autonomy level), latched
Calls /navigator/stop on the stop key, so stop also halts a route the navigator is walking.
Reads (from the robot, or the sim's ROS bridge):
  /g1/velocity         geometry_msgs/TwistStamped   measured velocity, body frame
  /g1/tilt             std_msgs/Float32             degrees from upright
"""

from __future__ import annotations

import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

# Training speed ranges (training/g1_comp/tasks.py, OPEN_SPEEDS). The policy has
# never seen anything outside them, so the operator can't send it.
LIMITS = {'x': (-0.5, 1.0), 'y': (-0.5, 0.5), 'yaw': (-0.5, 0.5)}  # walk_policy.COMMAND_RANGE
STEPS = {'x': 0.1, 'y': 0.1, 'yaw': 0.1}
KEYS = {'w': ('x', +1), 's': ('x', -1), 'a': ('y', +1), 'd': ('y', -1), 'q': ('yaw', +1), 'e': ('yaw', -1)}
STOP_KEYS = (' ', 'x')
RESET_KEY = 'r'
PUBLISH_HZ = 20.0
CMD_TOPIC = '/cmd_vel/teleop'   # cmd_vel_mux (ros2_ws/src/navigation) passes it on to /cmd_vel
LINK_TIMEOUT = 1.0  # s without robot status before the link counts as lost
FALLEN_TILT = 45.0  # degrees
HELP = """\
  w / s   forward / back        a / d   left / right        q / e   turn left / right
  space   stop                  r       stop, and stand a fallen robot up (sim testing only)
  Ctrl+C  stop and quit
"""


@dataclass
class Command:
    """The operator's velocity command. Pure logic, no ROS, so it can be tested."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def press(self, key: str) -> bool:
        """Apply one key press. Returns False if the key means nothing."""
        key = key.lower()
        if key in STOP_KEYS:
            self.x = self.y = self.yaw = 0.0
            return True
        if key not in KEYS:
            return False
        axis, sign = KEYS[key]
        lo, hi = LIMITS[axis]
        value = getattr(self, axis) + sign * STEPS[axis]
        value = round(min(max(value, lo), hi), 3)  # round: no 0.30000000000000004
        setattr(self, axis, 0.0 if value == 0 else value)  # no "-0.0" on the display
        return True


def status_line(cmd: Command, velocity, tilt, link_age, extra_publishers: int) -> str:
    """The one-line operator display."""
    parts = [f'MODE TELEOP | cmd x {cmd.x:+.1f} y {cmd.y:+.1f} turn {cmd.yaw:+.1f}']
    if link_age is None or link_age > LINK_TIMEOUT:
        parts.append('ROBOT LINK LOST')
    else:
        vx, vy, wz = velocity
        parts.append(f'actual x {vx:+.2f} y {vy:+.2f} turn {wz:+.2f}')
        parts.append('FALLEN' if tilt > FALLEN_TILT else f'tilt {tilt:4.1f} deg')
    if extra_publishers:
        parts.append(f'WARNING: {extra_publishers} other /cmd_vel source(s)')
    return ' | '.join(parts)


class ResetRequest:
    """Asks the sim to stand a fallen robot up where it is (/sim/respawn), and remembers the answer to show.

    Sim testing only: the service exists only in a scenario with on_fall: respawn, never on the real robot.
    """

    SERVICE = '/sim/respawn'
    SHOW_FOR = 6.0  # s the answer stays on the status line

    def __init__(self, node):
        from std_srvs.srv import Trigger
        self.trigger = Trigger
        self.client = node.create_client(Trigger, self.SERVICE)
        self.text, self.when = '', 0.0

    def request(self) -> None:
        if not self.client.service_is_ready():
            self._show('RESET unavailable: sim testing only (scenario room_teleop)')
            return
        self._show('RESET requested')
        self.client.call_async(self.trigger.Request()).add_done_callback(self._done)

    def _done(self, future) -> None:
        result = future.result()
        if result is None:
            self._show('RESET failed: no answer from the sim')
        else:
            self._show(('RESET: ' if result.success else 'RESET failed: ') + result.message)

    def _show(self, text: str) -> None:
        self.text, self.when = text, time.monotonic()

    def note(self) -> str:
        return self.text if self.text and time.monotonic() - self.when < self.SHOW_FOR else ''


class NavigatorStop:
    """Stops the navigator (/navigator/stop) when the operator presses stop, if it is running."""

    SERVICE = '/navigator/stop'

    def __init__(self, node):
        from std_srvs.srv import Trigger
        self.trigger = Trigger
        self.client = node.create_client(Trigger, self.SERVICE)

    def request(self) -> None:
        if self.client.service_is_ready():
            self.client.call_async(self.trigger.Request())


class RawTerminal:
    """Read single key presses without Enter, and restore the terminal on exit."""

    def __enter__(self):
        fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        mode = termios.tcgetattr(fd)
        mode[3] &= ~termios.ECHO  # Python 3.12's setcbreak no longer turns echo off
        termios.tcsetattr(fd, termios.TCSADRAIN, mode)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def key(self, timeout: float) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float32, String

    if not sys.stdin.isatty():
        sys.exit('keyboard_teleop needs a terminal: run it with `ros2 run`, not from a launch file.')

    # Our own Ctrl+C handling: rclpy's would shut ROS down before the final stop is sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('keyboard_teleop')
    cmd_pub = node.create_publisher(Twist, CMD_TOPIC, 10)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    mode_pub = node.create_publisher(String, '/g1/autonomy_mode', latched)
    mode_pub.publish(String(data='teleop'))

    state = {'velocity': (0.0, 0.0, 0.0), 'tilt': 0.0, 'seen': None}

    def on_velocity(msg):
        state['velocity'] = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)
        state['seen'] = time.monotonic()

    def on_tilt(msg):
        state['tilt'] = msg.data
        state['seen'] = time.monotonic()

    node.create_subscription(TwistStamped, '/g1/velocity', on_velocity, 10)
    node.create_subscription(Float32, '/g1/tilt', on_tilt, 10)
    reset = ResetRequest(node)
    nav_stop = NavigatorStop(node)

    cmd = Command()

    def publish(c: Command) -> None:
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = c.x, c.y, c.yaw
        cmd_pub.publish(msg)

    def on_term(*_):
        raise KeyboardInterrupt  # killed or terminal closed: still stop the robot on the way out

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)

    # Robot status arrives faster than we redraw: take it in on its own thread so
    # the display never falls behind.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    print(HELP)
    period = 1.0 / PUBLISH_HZ
    try:
        with RawTerminal() as term:
            next_publish = time.monotonic()
            while rclpy.ok():
                key = term.key(timeout=max(0.0, next_publish - time.monotonic()))
                if key == '\x03':  # Ctrl+C in cbreak mode arrives as SIGINT, but be safe
                    break
                if key is not None and key.lower() == RESET_KEY:
                    cmd.press(' ')  # stop first: the robot comes back standing still
                    reset.request()
                elif key is not None:
                    cmd.press(key)
                    if key in STOP_KEYS:
                        nav_stop.request()
                if time.monotonic() >= next_publish:
                    next_publish += period
                    publish(cmd)
                    seen = state['seen']
                    age = None if seen is None else time.monotonic() - seen
                    others = max(0, node.count_publishers(CMD_TOPIC) - 1)
                    line = status_line(cmd, state['velocity'], state['tilt'], age, others)
                    if reset.note():
                        line += ' | ' + reset.note()
                    sys.stdout.write('\r\033[K' + line)
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        # Leave the robot stopped: a few zero commands in case one is lost.
        for _ in range(5):
            publish(Command())
            time.sleep(0.02)
        print('\nStopped.')
        executor.shutdown()
        spinner.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
