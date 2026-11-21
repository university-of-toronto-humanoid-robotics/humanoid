from teleop.gamepad_teleop import DEFAULTS, JOY_TIMEOUT, Gamepad, scale, sticks_to_command
from teleop.keyboard_teleop import LIMITS

START, STOP = DEFAULTS['button_start'], DEFAULTS['button_stop']


def axes(forward=0.0, sideways=0.0, turn=0.0):
    a = [0.0] * 8
    a[DEFAULTS['axis_forward']], a[DEFAULTS['axis_sideways']], a[DEFAULTS['axis_turn']] = forward, sideways, turn
    return a


def buttons(*pressed):
    return [1 if i in pressed else 0 for i in range(13)]


def test_scale_dead_zone_and_limits():
    assert scale(0.0, (-0.5, 1.0)) == 0.0
    assert scale(0.09, (-0.5, 1.0)) == 0.0  # a resting stick is never exactly 0
    assert scale(1.0, (-0.5, 1.0)) == 1.0
    assert scale(-1.0, (-0.5, 1.0)) == -0.5  # backward is slower than forward
    assert scale(5.0, (-0.5, 1.0)) == 1.0  # bad input can't exceed the limits
    assert 0.0 < scale(0.5, (-0.5, 1.0)) < 1.0  # proportional in between


def test_sticks_follow_ros_directions():
    m = DEFAULTS
    cmd = sticks_to_command(axes(forward=1.0, sideways=1.0, turn=-1.0), m['axis_forward'], m['axis_sideways'], m['axis_turn'])
    assert cmd.x == LIMITS['x'][1]  # stick up: forward
    assert cmd.y == LIMITS['y'][1]  # stick left: step left
    assert cmd.yaw == LIMITS['yaw'][0]  # stick right: turn right


def test_missing_axes_are_zero():
    cmd = sticks_to_command([0.5], 1, 0, 3)  # a controller with fewer axes than expected
    assert (cmd.x, cmd.yaw) == (0.0, 0.0)


def test_starts_locked_until_start_pressed():
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(forward=1.0), buttons(), now=0.0)
    assert pad.command(0.0).x == 0.0 and 'LOCKED' in pad.note(0.0)
    pad.update(axes(), buttons(START), now=0.1)  # unlock with the sticks let go
    pad.update(axes(forward=1.0), buttons(), now=0.2)
    assert pad.command(0.2).x == LIMITS['x'][1] and 'DRIVING' in pad.note(0.2)


def test_unlock_needs_centred_sticks():
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(turn=1.0), buttons(START), now=0.0)  # pushed stick, or a wrong axis resting at +1
    assert pad.locked and pad.command(0.0).yaw == 0.0
    assert 'let go of the sticks' in pad.note(0.0)
    pad.update(axes(), buttons(START), now=0.1)
    assert not pad.locked


def test_stop_button_locks_and_wins_over_start():
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(), buttons(START), now=0.0)
    pad.update(axes(forward=1.0), buttons(STOP), now=0.1)
    assert pad.command(0.1).x == 0.0
    pad.update(axes(forward=1.0), buttons(), now=0.2)  # stays locked after release
    assert pad.command(0.2).x == 0.0
    pad.update(axes(forward=1.0), buttons(START, STOP), now=0.3)
    assert pad.command(0.3).x == 0.0


def test_releasing_sticks_stops():
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(), buttons(START), now=0.0)
    pad.update(axes(forward=1.0), buttons(), now=0.05)
    assert pad.command(0.05).x > 0
    pad.update(axes(), buttons(), now=0.1)
    c = pad.command(0.1)
    assert (c.x, c.y, c.yaw) == (0.0, 0.0, 0.0)


def test_silent_controller_stops_the_robot():
    pad = Gamepad(dict(DEFAULTS))
    assert 'NO CONTROLLER' in pad.note(0.0)  # before any message
    pad.update(axes(), buttons(START), now=0.0)
    pad.update(axes(forward=1.0), buttons(), now=0.0)
    assert pad.command(JOY_TIMEOUT - 0.1).x > 0
    late = JOY_TIMEOUT + 0.1
    assert pad.command(late).x == 0.0 and 'NO CONTROLLER' in pad.note(late)


def test_reconnect_stays_locked_until_start():
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(), buttons(START), now=0.0)
    pad.update(axes(forward=1.0), buttons(), now=0.1)
    back = JOY_TIMEOUT + 1.0
    pad.update(axes(forward=1.0), buttons(), now=back)  # back with the stick still pushed
    assert pad.command(back).x == 0.0 and 'LOCKED' in pad.note(back)
    pad.update(axes(), buttons(START), now=back + 0.1)
    pad.update(axes(forward=1.0), buttons(), now=back + 0.2)
    assert pad.command(back + 0.2).x == LIMITS['x'][1]


def test_reset_button_locks_and_fires_once_per_press():
    reset = DEFAULTS['button_reset']
    pad = Gamepad(dict(DEFAULTS))
    pad.update(axes(), buttons(START), now=0.0)
    assert not pad.locked
    pad.update(axes(), buttons(reset), now=0.1)
    assert pad.locked and pad.take_reset()
    assert not pad.take_reset()  # taken
    pad.update(axes(), buttons(reset), now=0.2)  # still held: no second reset
    assert not pad.take_reset()
    pad.update(axes(), buttons(), now=0.3)
    pad.update(axes(), buttons(reset), now=0.4)  # pressed again
    assert pad.take_reset()
