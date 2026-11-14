import math

import numpy as np
import pytest

from height_scan.scan import (N_ACROSS, N_ALONG, N_POINTS, NEAR, OFFSETS, compute_scan, sample, scan_points,
                              unpack_layer)

RES = 0.05
N = 100                       # cells per side: a 5 m map
TOP = (N * RES / 2,) * 2      # +x, +y corner of a map centred on the origin
Z = 0.78                      # pelvis height


def world_grid(height_at):
    """A grid_map layer centred on the origin: row 0 is the largest x, column 0 the largest y."""
    centres = TOP[0] - (np.arange(N) + 0.5) * RES
    return height_at(*np.meshgrid(centres, centres, indexing='ij')).astype(float)


def scan_of(observed, filled=None, yaw=0.0, feet_ground=0.0):
    return compute_scan(observed, filled, TOP, RES, (0.0, 0.0, Z), yaw, feet_ground)


def test_shape_and_order():
    assert (N_ALONG, N_ACROSS, N_POINTS) == (33, 21, 693)
    assert OFFSETS[0] == pytest.approx([-0.5, -0.5])    # back right
    assert OFFSETS[1] == pytest.approx([-0.45, -0.5])   # along the heading is the inner loop
    assert OFFSETS[33] == pytest.approx([-0.5, -0.45])  # across it, the outer one
    assert OFFSETS[-1] == pytest.approx([1.1, 0.5])     # front left


def test_points_turn_with_yaw():
    assert scan_points((1.0, 2.0), math.pi / 2)[-1] == pytest.approx([1.0 - 0.5, 2.0 + 1.1])


def test_unpack_layer_undoes_column_order_and_circular_buffer():
    want = np.arange(12.0).reshape(3, 4)                # 3 rows (x), 4 columns (y)
    shifted = np.roll(want, (1, 2), axis=(0, 1))        # the map moved: storage starts at cell (1, 2)
    flat = shifted.T.ravel()                            # column by column
    assert unpack_layer(flat, cols=4, rows=3, outer_start=1, inner_start=2) == pytest.approx(want)


def test_sample_finds_the_right_cell_and_nan_off_the_map():
    layer = world_grid(lambda x, y: x + 10 * y)
    values = sample(layer, TOP, RES, np.array([[1.012, 0.512], [-0.387, 0.113], [2.6, 0.0]]))
    assert values[:2] == pytest.approx([1.025 + 5.25, -0.375 + 1.25])
    assert math.isnan(values[2])


def test_flat_floor_reads_pelvis_height():
    scan, observed, from_feet = scan_of(np.zeros((N, N)))
    assert scan == pytest.approx(np.full(N_POINTS, Z))
    assert (observed, from_feet) == (1.0, 0.0)


def test_step_ahead_appears_ahead_and_turns_with_yaw():
    along = OFFSETS[:, 0]
    for yaw, step in [(0.0, lambda x, y: 0.2 * (x > 0.5)), (math.pi / 2, lambda x, y: 0.2 * (y > 0.5))]:
        scan, *_ = scan_of(world_grid(step), yaw=yaw)
        assert scan[along > 0.55] == pytest.approx(Z - 0.2)
        assert scan[along < 0.45] == pytest.approx(Z)


def test_tall_wall_and_deep_drop_are_clipped():
    assert scan_of(np.full((N, N), 5.0))[0].min() == -1.0
    assert scan_of(np.full((N, N), -5.0))[0].max() == 2.0


def test_unseen_ground_near_comes_from_the_feet_and_far_from_inpainting():
    unseen = np.full((N, N), np.nan)
    scan, observed, from_feet = scan_of(unseen, filled=np.full((N, N), 0.3), feet_ground=0.1)
    near = np.hypot(*OFFSETS.T) < NEAR - 1e-6   # points exactly NEAR away count as far
    assert scan[near] == pytest.approx(Z - 0.1)
    assert scan[~near] == pytest.approx(Z - 0.3)
    assert (observed, from_feet) == (0.0, pytest.approx(near.mean()))


def test_unseen_ground_without_inpainting_comes_from_the_feet():
    scan, _, from_feet = scan_of(np.full((N, N), np.nan), filled=None, feet_ground=0.1)
    assert scan == pytest.approx(np.full(N_POINTS, Z - 0.1))
    assert from_feet == 1.0


def test_observed_ground_wins_over_the_feet():
    scan, observed, _ = scan_of(np.full((N, N), 0.2), filled=np.zeros((N, N)), feet_ground=0.0)
    assert scan == pytest.approx(np.full(N_POINTS, Z - 0.2))
    assert observed == 1.0
