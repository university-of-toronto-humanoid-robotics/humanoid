from types import SimpleNamespace as NS

import pytest

pytest.importorskip('geometry_msgs')   # the mux module imports ROS messages

from navigation.cmd_vel_mux import Mux  # noqa: E402


def twist(x=0.0, yaw=0.0):
    return NS(linear=NS(x=x, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=yaw))


def test_navigator_drives_while_the_operator_sends_zeros():
    mux = Mux(hold=1.0)
    for k in range(20):
        assert mux.teleop(twist(), k * 0.05) == (False, False)   # locked teleop: zeros, not forwarded
        assert mux.nav(twist(0.4), k * 0.05)


def test_a_stick_takes_over_once_and_blocks_the_navigator():
    mux = Mux(hold=1.0)
    assert mux.teleop(twist(0.3), 0.0) == (True, True)
    assert mux.teleop(twist(0.3), 0.05) == (True, False)      # still driving: no second takeover
    assert not mux.nav(twist(0.4), 0.1)
    assert mux.teleop(twist(), 0.5) == (True, False)          # letting go: the zeros reach the robot
    assert not mux.nav(twist(0.4), 0.9)


def test_control_returns_after_hold_seconds_of_zeros():
    mux = Mux(hold=1.0)
    mux.teleop(twist(0.3), 0.0)
    assert not mux.nav(twist(0.4), 0.99)
    assert mux.nav(twist(0.4), 1.01)
    assert mux.teleop(twist(), 1.05) == (False, False)
    assert mux.teleop(twist(yaw=0.2), 2.0) == (True, True)    # a new takeover
