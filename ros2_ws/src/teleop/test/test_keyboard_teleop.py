from teleop.keyboard_teleop import LIMITS, Command, status_line


def test_keys_nudge_and_stop():
    cmd = Command()
    for key in 'wwwaq':
        assert cmd.press(key)
    assert (cmd.x, cmd.y, cmd.yaw) == (0.3, 0.1, 0.2)
    assert cmd.press(' ')
    assert (cmd.x, cmd.y, cmd.yaw) == (0.0, 0.0, 0.0)


def test_unknown_key_does_nothing():
    cmd = Command(x=0.5)
    assert not cmd.press('z')
    assert cmd.x == 0.5


def test_clamped_to_training_ranges():
    cmd = Command()
    for _ in range(50):
        cmd.press('w')
        cmd.press('d')
        cmd.press('q')
    assert cmd.x == LIMITS['x'][1]
    assert cmd.y == LIMITS['y'][0]
    assert cmd.yaw == LIMITS['yaw'][1]
    for _ in range(50):
        cmd.press('s')
    assert cmd.x == LIMITS['x'][0]


def test_no_negative_zero():
    cmd = Command()
    cmd.press('w')
    cmd.press('s')
    assert '-0.0' not in status_line(cmd, (0, 0, 0), 0.0, 0.1, 0)


def test_status_line():
    cmd = Command(x=0.5)
    assert 'ROBOT LINK LOST' in status_line(cmd, (0, 0, 0), 0.0, None, 0)
    assert 'ROBOT LINK LOST' in status_line(cmd, (0, 0, 0), 0.0, 5.0, 0)
    ok = status_line(cmd, (0.48, 0.0, 0.0), 3.0, 0.1, 0)
    assert 'actual x +0.48' in ok and 'tilt' in ok
    assert 'FALLEN' in status_line(cmd, (0, 0, 0), 80.0, 0.1, 0)
    assert 'other /cmd_vel' in status_line(cmd, (0, 0, 0), 0.0, 0.1, 1)
