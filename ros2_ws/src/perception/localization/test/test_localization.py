from pathlib import Path

import numpy as np
import pytest

from localization.course_map import build_map, ground_offset, height_at, load_grid
from localization.icp import CourseMatcher, Pose, wrap

GRID = next(p / 'sim' / 'competition' / 'course_heightmap.npz' for p in Path(__file__).resolve().parents
            if (p / 'sim' / 'competition' / 'course_heightmap.npz').is_file())


@pytest.fixture(scope='module')
def course():
    grid = load_grid(GRID)
    points, normals = build_map(*grid)
    return grid, CourseMatcher(points, normals)


def fake_scan(matcher, robot_xy, odom_pose, seed=0, reach=6.0, clutter=300):
    """What the LiDAR would see near robot_xy, as points in odom: map points plus noise, plus things not in the map."""
    rng = np.random.default_rng(seed)
    near = np.linalg.norm(matcher.points[:, :2] - robot_xy, axis=1) < reach
    seen = matcher.points[near][rng.random(near.sum()) < 0.5]
    seen = seen + rng.normal(0, 0.02, seen.shape)
    junk = np.column_stack([robot_xy + rng.uniform(-3, 3, (clutter, 2)), rng.uniform(0, 1.5, clutter)])
    return odom_pose.inverse().apply(np.concatenate([seen, junk]))


def test_map_has_floor_and_walls(course):
    _, matcher = course
    vertical = np.abs(matcher.normals[:, 2]) > 0.9
    horizontal = np.abs(matcher.normals[:, 2]) < 0.1
    assert vertical.sum() > 5000 and horizontal.sum() > 5000
    assert np.allclose(np.linalg.norm(matcher.normals, axis=1), 1.0)


def test_pose_algebra():
    a, b = Pose(1.0, 2.0, 0.3, 0.5), Pose(-0.4, 0.7, 0.1, -1.2)
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(a.compose(b).apply(p), a.apply(b.apply(p)))
    assert np.allclose(a.inverse().apply(a.apply(p)), p)


@pytest.mark.parametrize('where', [(10.5, 4.2), (7.3, 4.2), (4.5, 4.0), (9.0, 3.5)])
def test_align_from_close_guess(course, where):
    _, matcher = course
    truth = Pose(where[0], where[1], 0.9, 2.8)
    scan = fake_scan(matcher, np.array(where), truth)
    r = matcher.align(scan, Pose(truth.x + 0.1, truth.y - 0.08, truth.z + 0.05, truth.yaw + np.radians(3)))
    xy, yaw = r.pose.distance(truth)
    assert xy < 0.03 and yaw < np.radians(0.5) and abs(r.pose.z - truth.z) < 0.02
    assert r.fitness > 0.7


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_search_from_rough_guess(course, seed):
    grid, matcher = course
    rng = np.random.default_rng(seed)
    where = np.array([9.0 + rng.uniform(-1, 2), 4.2 + rng.uniform(-0.4, 0.4)])
    truth = Pose(where[0], where[1], 0.8, wrap(np.pi + rng.uniform(-0.5, 0.5)))
    scan = fake_scan(matcher, where, truth, seed=seed)
    guess = Pose(truth.x + rng.uniform(-0.4, 0.4), truth.y + rng.uniform(-0.4, 0.4), 0.0,
                 wrap(truth.yaw + np.radians(rng.uniform(-15, 15))))
    guess.z = ground_offset(*grid, guess.apply(scan))
    r = matcher.search(scan, guess)
    xy, yaw = r.pose.distance(truth)
    assert xy < 0.05 and yaw < np.radians(1.0)


def test_bad_guess_scores_low(course):
    _, matcher = course
    truth = Pose(10.5, 4.2, 0.9, 0.0)
    scan = fake_scan(matcher, np.array([10.5, 4.2]), truth)
    good, _ = matcher.score(scan, truth)
    bad, _ = matcher.score(scan, Pose(10.5, 4.2, 0.9, np.radians(90)))
    assert good > 0.7 and bad < good - 0.3


def test_height_at_outside_is_nan(course):
    grid, _ = course
    assert np.isnan(height_at(*grid, np.array([-50.0]), np.array([0.0])))[0]


def test_points_beyond_the_course_dont_count(course):
    _, matcher = course
    truth = Pose(10.5, 4.2, 0.9, 0.0)
    scan = fake_scan(matcher, np.array([10.5, 4.2]), truth, clutter=0)
    rng = np.random.default_rng(3)
    beyond = np.column_stack([rng.uniform(10, 12, 3000), rng.uniform(8, 20, 3000), np.zeros(3000)])  # floor off the side
    with_beyond = np.concatenate([scan, truth.inverse().apply(beyond)])
    assert matcher.score(with_beyond, truth)[0] == pytest.approx(matcher.score(scan, truth)[0])
