from pathlib import Path

import numpy as np
import pytest

from navigation.follower import MAX_TURN, Follower, wrap
from navigation.route import load_route, make_route, obstacles_near

ROUTES = Path(__file__).resolve().parents[1] / 'routes'
GRID = next(p / 'sim' / 'competition' / 'course_heightmap.npz' for p in Path(__file__).resolve().parents
            if (p / 'sim' / 'competition' / 'course_heightmap.npz').is_file())
DT = 0.05


def walk(route, start, yaw, seconds=120.0, lag=0.3, noise=0.0, seed=0, deadband=0.0, gain=(1.0, 1.0, 1.0)):
    """A toy robot: it tracks the command with a lag (s) and some noise, and stands still for commands whose
    |vx| + |vy| + |wz| is under deadband, as the policies do. gain scales how much of each it actually does, and the
    deadband applies to what it does.
    Returns the follower and the trajectory."""
    rng = np.random.default_rng(seed)
    xy, velocity = np.array(start, dtype=float), np.zeros(3)
    follower = Follower(route, xy)
    path, commands = [], []
    for k in range(int(seconds / DT)):
        command = follower.update(k * DT, xy, yaw)
        commands.append(command)
        wanted = command * gain
        wanted = wanted if np.abs(wanted).sum() >= deadband else np.zeros(3)
        velocity += (wanted - velocity) * DT / lag + rng.normal(0, noise, 3)
        c, s = np.cos(yaw), np.sin(yaw)
        xy = xy + DT * np.array([c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1]])
        yaw = wrap(yaw + DT * velocity[2])
        path.append((xy.copy(), yaw, follower.state, follower.index))
        if follower.done:
            break
    return follower, path, np.array(commands)


def test_reaches_the_last_stop_facing_its_heading():
    route = make_route({'speed': 0.4, 'waypoints': [{'x': 2, 'y': 0}, {'x': 2, 'y': 1.5, 'yaw': 180}]})
    follower, path, commands = walk(route, (0, 0), 0.0)
    assert follower.done
    xy, yaw, _, _ = path[-1]
    assert np.linalg.norm(xy - [2, 1.5]) < 0.1
    assert abs(wrap(yaw - np.pi)) < np.radians(5)
    assert np.all(np.abs(commands[:, 2]) <= MAX_TURN + 1e-9)


def test_walks_through_a_waypoint_without_stopping():
    route = make_route({'speed': 0.4, 'waypoints': [{'x': 2, 'y': 0}, {'x': 2, 'y': 2}]})
    follower, path, commands = walk(route, (0, 0), 0.0)
    assert follower.done
    # While walking past the corner it never stops, and it passes within 0.3 m of the corner.
    corner = [i for i, p in enumerate(path) if np.linalg.norm(p[0] - [2, 0]) < 0.6]
    assert min(np.hypot(*commands[i, :2]) for i in corner) > 0.1
    assert min(np.linalg.norm(p[0] - [2, 0]) for p in path) < 0.3


def test_turns_on_the_spot_when_facing_away():
    route = make_route({'waypoints': [{'x': -2, 'y': 0}]})
    follower = Follower(route, (0, 0))
    command = follower.update(0.0, (0, 0), 0.0)
    for k in range(1, 20):
        command = follower.update(k * DT, (0, 0), 0.0)
    assert command[0] == 0.0 and abs(command[2]) == pytest.approx(MAX_TURN)


def test_stops_and_waits():
    route = make_route({'waypoints': [{'x': 1, 'y': 0, 'wait': 2.0}, {'x': 2, 'y': 0}]})
    follower, path, _ = walk(route, (0, 0), 0.0)
    waited = sum(1 for p in path if p[2] == 'waiting') * DT
    assert follower.done and waited == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize('seed', range(3))
def test_noisy_robot_still_arrives(seed):
    route = make_route({'speed': 0.5, 'waypoints': [{'x': 1.5, 'y': 0.5}, {'x': 3, 'y': -0.5},
                                                   {'x': 3, 'y': 1, 'yaw': 90, 'tolerance': 0.05}]})
    follower, path, _ = walk(route, (0, 0), 0.3, noise=0.01, seed=seed)
    assert follower.done
    assert np.linalg.norm(path[-1][0] - [3, 1]) < 0.05


def grid():
    d = np.load(GRID)
    return d['heights'], float(d['x0']), float(d['y0']), float(d['res'])


@pytest.mark.parametrize('path', sorted(ROUTES.glob('*.yaml')), ids=lambda p: p.stem)
def test_route_files_load_and_clear_the_course(path):
    route = load_route(path)
    assert route.waypoints and route.waypoints[-1].stop
    hits = obstacles_near(route, *grid())
    assert not hits, f'{path.name} passes within 0.3 m of something tall, first at {hits[0]}'


def test_obstacle_check_finds_the_door_leaf():
    # The first version of start_area_loop ended here, pressed against the open door leaf.
    route = make_route({'waypoints': [{'x': 11.0, 'y': 4.2}, {'x': 10.4, 'y': 4.2}]})
    assert obstacles_near(route, *grid())


def test_lines_up_despite_the_policys_deadband():
    """The policies stand still for commands under 0.1, so lining up must ask for more than that."""
    route = make_route({'waypoints': [{'x': 2, 'y': 0.3, 'yaw': 180, 'tolerance': 0.05}]})
    follower, path, _ = walk(route, (0, 0), 0.0, deadband=0.1, seconds=60.0)
    assert follower.done and not follower.loose
    assert np.linalg.norm(path[-1][0] - [2, 0.3]) < 0.05


def test_pushes_harder_when_the_robot_barely_responds():
    """What the sim showed: slow forward steps and turns one way move the robot only a fraction as far."""
    route = make_route({'waypoints': [{'x': 0.3, 'y': -0.2, 'yaw': -60, 'tolerance': 0.05}]})
    follower, path, _ = walk(route, (0, 0), 0.0, deadband=0.1, gain=(0.3, 1.0, 0.3), seconds=60.0)
    assert follower.done
    assert np.linalg.norm(path[-1][0] - [0.3, -0.2]) < 0.05
