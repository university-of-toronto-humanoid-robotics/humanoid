"""PS4 controller teleoperation for the G1, laid out like Unitree's own remote.

Run the controller driver in one terminal and this in another:

  ros2 run joy game_controller_node
  ros2 run teleop gamepad_teleop

Controls (as on the Unitree remote: left stick walks, right stick turns):
  left stick up / down      forward / back
  left stick left / right   step left / right
  right stick left / right  turn left / right
  OPTIONS                   start: unlock driving (Unitree: START)
  CIRCLE                    stop and lock (Unitree: L2 + B); also stops the navigator
  SHARE                     stop, lock, and stand a fallen robot up (sim testing only)

It starts locked, so the robot stands still until OPTIONS is pressed with the
sticks let go. Letting go
of the sticks stops the robot. Speeds are proportional to the stick and limited
to what the policy was trained on.

Same topics, status line and safety as keyboard_teleop. The robot also stops if
the controller goes quiet (unplugged, out of range, joy_node closed), and stays
locked when it comes back until OPTIONS is pressed again.

Button and axis numbers are game_controller_node's standard layout. With
joy_node instead, they depend on the driver: check with `ros2 topic echo /joy`
and change them with parameters, e.g.
  ros2 run teleop gamepad_teleop --ros-args -p axis_turn:=3
"""

from __future__ import annotations

import signal
import sys
import threading
import time

from teleop.keyboard_teleop import CMD_TOPIC, LIMITS, PUBLISH_HZ, Command, NavigatorStop, ResetRequest, status_line

# joy's game_controller_node, which gives every recognised controller (PS4, Xbox,
# ...) the same standard layout whatever the driver. Left and up are positive, as
# ROS expects. Measured on a DualShock 4 (WSL, USB).
DEFAULTS = {
    'axis_forward': 1,  # left stick up/down
    'axis_sideways': 0,  # left stick left/right
    'axis_turn': 2,  # right stick left/right
    'button_start': 6,  # OPTIONS (Xbox: Menu)
    'button_stop': 1,  # CIRCLE (Xbox: B)
    'button_reset': 4,  # SHARE (Xbox: View)
}
DEADZONE = 0.1  # stick travel ignored around the centre, so a resting stick is zero
JOY_TIMEOUT = 0.5  # s without a controller message before the robot is stopped


def scale(stick: float, limits: tuple[float, float]) -> float:
    """Stick position (-1..1) to a speed inside `limits`, with a dead zone.

    The two directions can have different limits (the robot walks forward
    faster than backward), so each half of the stick is scaled to its own.
    """
    stick = min(max(stick, -1.0), 1.0)
    if abs(stick) < DEADZONE:
        return 0.0
    travel = (abs(stick) - DEADZONE) / (1.0 - DEADZONE)  # 0 at the dead zone's edge, 1 at full
    lo, hi = limits
    return round(travel * (hi if stick > 0 else lo), 3)


def sticks_to_command(axes, forward: int, sideways: int, turn: int) -> Command:
    """The velocity command for the current stick positions. Pure logic, no ROS."""

    def axis(i: int) -> float:
        return axes[i] if 0 <= i < len(axes) else 0.0

    return Command(
        x=scale(axis(forward), LIMITS['x']),
        y=scale(axis(sideways), LIMITS['y']),
        yaw=scale(axis(turn), LIMITS['yaw']),
    )


class Gamepad:
    """Turns controller messages into a command, with the lock and the timeout."""

    def __init__(self, mapping: dict[str, int]):
        self.map = mapping
        self.locked = True  # stand still until START
        self.axes: list[float] = []
        self.seen: float | None = None
        self.reset_held = False
        self.reset_wanted = False
        self.stop_held = False
        self.stop_wanted = False

    def update(self, axes, buttons, now: float) -> None:
        def pressed(name: str) -> bool:
            i = self.map[name]
            return 0 <= i < len(buttons) and buttons[i] == 1

        if not self.connected(now):
            self.locked = True  # after a dropout, OPTIONS again before driving
        self.axes, self.seen = list(axes), now
        if pressed('button_reset') and not self.reset_held:  # once per press, however long it's held
            self.reset_wanted = True
            self.locked = True  # the robot comes back standing still: OPTIONS again to drive
        self.reset_held = pressed('button_reset')
        if pressed('button_stop') and not self.stop_held:  # once per press: also stops the navigator
            self.stop_wanted = True
        self.stop_held = pressed('button_stop')
        if pressed('button_stop'):  # stop wins if both are held
            self.locked = True
        elif pressed('button_start') and self.centred():
            # Only with the sticks let go, so unlocking never starts a move by
            # itself (a pushed stick, or a wrong axis number resting off zero).
            self.locked = False

    def take_stop(self) -> bool:
        """Whether CIRCLE was pressed since the last call."""
        wanted, self.stop_wanted = self.stop_wanted, False
        return wanted

    def take_reset(self) -> bool:
        """Whether SHARE was pressed since the last call."""
        wanted, self.reset_wanted = self.reset_wanted, False
        return wanted

    def centred(self) -> bool:
        m = self.map
        c = sticks_to_command(self.axes, m['axis_forward'], m['axis_sideways'], m['axis_turn'])
        return (c.x, c.y, c.yaw) == (0.0, 0.0, 0.0)

    def connected(self, now: float) -> bool:
        return self.seen is not None and now - self.seen < JOY_TIMEOUT

    def command(self, now: float) -> Command:
        if self.locked or not self.connected(now):
            return Command()
        m = self.map
        return sticks_to_command(self.axes, m['axis_forward'], m['axis_sideways'], m['axis_turn'])

    def note(self, now: float) -> str:
        if not self.connected(now):
            return 'NO CONTROLLER (is game_controller_node running?)'
        if self.locked:
            return 'LOCKED (press OPTIONS to drive)' if self.centred() else 'LOCKED (let go of the sticks, then OPTIONS)'
        return 'DRIVING (CIRCLE to stop)'


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Joy
    from std_msgs.msg import Float32, String

    # Our own Ctrl+C handling: rclpy's would shut ROS down before the final stop is sent.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('gamepad_teleop')
    pad = Gamepad({name: node.declare_parameter(name, value).value for name, value in DEFAULTS.items()})

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
    node.create_subscription(Joy, '/joy', lambda m: pad.update(m.axes, m.buttons, time.monotonic()), 10)

    def publish(c: Command) -> None:
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = c.x, c.y, c.yaw
        cmd_pub.publish(msg)

    def on_term(*_):
        raise KeyboardInterrupt  # Ctrl+C, killed or terminal closed: stop the robot on the way out

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_term)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    print('  left stick: walk    right stick: turn    OPTIONS: start    CIRCLE: stop    SHARE: reset    Ctrl+C: quit\n')
    show = sys.stdout.isatty()
    period = 1.0 / PUBLISH_HZ
    try:
        next_publish = time.monotonic()
        while rclpy.ok():
            time.sleep(max(0.0, next_publish - time.monotonic()))
            next_publish += period
            now = time.monotonic()
            if pad.take_reset():
                reset.request()
            if pad.take_stop():
                nav_stop.request()
            cmd = pad.command(now)
            publish(cmd)
            if show:
                age = None if state['seen'] is None else now - state['seen']
                others = max(0, node.count_publishers(CMD_TOPIC) - 1)
                line = status_line(cmd, state['velocity'], state['tilt'], age, others)
                sys.stdout.write('\r\033[K' + line + ' | ' + (reset.note() or pad.note(now)))
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
