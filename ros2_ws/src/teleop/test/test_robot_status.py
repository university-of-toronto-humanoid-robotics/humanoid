import math

from teleop.robot_status import SpeedEstimator, tilt_deg, yaw_of


def quat_about_x(deg):
    half = math.radians(deg) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)   # (w, x, y, z)


def test_tilt():
    assert tilt_deg((1.0, 0.0, 0.0, 0.0)) == 0.0
    assert math.isclose(tilt_deg(quat_about_x(30)), 30.0, abs_tol=1e-6)
    assert math.isclose(tilt_deg(quat_about_x(90)), 90.0, abs_tol=1e-6)   # lying on its side: FALLEN
    # Turning in place is not tilting.
    w, z = math.cos(math.radians(40)), math.sin(math.radians(40))
    assert math.isclose(tilt_deg((w, 0.0, 0.0, z)), 0.0, abs_tol=1e-6)


def test_yaw():
    half = math.radians(60) / 2
    assert math.isclose(yaw_of(0.0, 0.0, math.sin(half), math.cos(half)), math.radians(60))


def test_speed_in_heading_frame():
    est = SpeedEstimator(window=0.3)
    assert est.speed() is None
    # Walking along the world's +y while facing +y: that is forward for the robot.
    for i in range(5):
        est.add(i * 0.1, 0.0, 0.5 * i * 0.1, math.pi / 2)
    vx, vy = est.speed()
    assert math.isclose(vx, 0.5, abs_tol=1e-9) and math.isclose(vy, 0.0, abs_tol=1e-9)


def test_speed_uses_recent_window_only():
    est = SpeedEstimator(window=0.3)
    for i in range(10):                       # 1 s standing still...
        est.add(i * 0.1, 0.0, 0.0, 0.0)
    for i in range(10, 15):                   # ...then walking at 0.4 m/s
        est.add(i * 0.1, 0.4 * (i - 9) * 0.1, 0.0, 0.0)
    vx, _ = est.speed()
    assert math.isclose(vx, 0.4, rel_tol=1e-6)


def test_repeated_and_backwards_stamps():
    est = SpeedEstimator()
    est.add(1.0, 0.0, 0.0, 0.0)
    est.add(1.0, 5.0, 0.0, 0.0)               # same stamp again: ignored
    est.add(1.1, 0.1, 0.0, 0.0)
    assert math.isclose(est.speed()[0], 1.0)
    est.add(0.5, 0.0, 0.0, 0.0)               # sim restarted: history dropped
    assert est.speed() is None
