"""The terrain scan the team's height-map policies read, built from an elevation map. No ROS in here.

Used by height_scan_node (robot and sim) and by bridge's TrueTerrainScan (sim only), so both share one definition.

The pattern is training/g1_comp/sensors.py's terrain_scan, and it must match it exactly:
  - 33 x 21 = 693 points, 5 cm apart: 1.6 m along the pelvis heading by 1.0 m across, centred 0.3 m ahead of the
    pelvis (so 0.5 m behind to 1.1 m ahead), turning with the pelvis yaw only.
  - Order: mjlab's GridPatternCfg. The 21 rows across the heading (right to left) are the outer loop, and the 33
    points along it (back to front) the inner one. scan[0] is back right, scan[1] 5 cm ahead of it, scan[33]
    5 cm to the left of it.
  - Each value is the pelvis height above the ground at that point, in metres, clipped to [-1, 2]. A point with
    no ground reads 2.
"""
import numpy as np

SIZE = (1.6, 1.0)           # m, along the heading x across it
RESOLUTION = 0.05           # m
FORWARD = 0.3               # m, grid centre ahead of the pelvis
CLIP = (-1.0, 2.0)          # m; -1 is "too tall to climb", 2 is "deep drop" or nothing there
STANDING_HEIGHT = 0.78      # m, pelvis above flat ground when standing
NEAR = 0.5                  # m, the zone under the robot where unseen ground is taken from the feet

_along = np.arange(-SIZE[0] / 2, SIZE[0] / 2 + RESOLUTION / 2, RESOLUTION)
_across = np.arange(-SIZE[1] / 2, SIZE[1] / 2 + RESOLUTION / 2, RESOLUTION)
_gx, _gy = np.meshgrid(_along, _across, indexing='xy')
OFFSETS = np.stack([_gx.ravel() + FORWARD, _gy.ravel()], axis=1)    # (693, 2) in the pelvis's yaw frame
N_ALONG, N_ACROSS = len(_along), len(_across)                       # 33, 21
N_POINTS = len(OFFSETS)                                             # 693
# Points within NEAR of the pelvis. Several lie exactly NEAR away; the margin keeps them out at every yaw.
_NEAR_MASK = np.hypot(*OFFSETS.T) < NEAR - 1e-6


def scan_points(xy, yaw):
    """World (x, y) of each scan point for a pelvis at xy with heading yaw. (693, 2)"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(xy) + OFFSETS @ np.array([[c, s], [-s, c]])


def to_scan(pelvis_z, ground):
    """Scan values from ground heights (NaN where there is no ground)."""
    return np.clip(np.where(np.isfinite(ground), pelvis_z - ground, CLIP[1]), *CLIP)


def unpack_layer(data, cols, rows, outer_start=0, inner_start=0):
    """A grid_map layer's flat data as a (rows, cols) array, row 0 at the map's +x edge.

    grid_map stores a layer column by column, in a circular buffer whose first cell is at (outer_start,
    inner_start) as the map moves; this undoes both.
    """
    a = np.asarray(data, dtype=float).reshape(cols, rows).T
    return np.roll(a, (-outer_start, -inner_start), axis=(0, 1))


def sample(layer, top_xy, resolution, points):
    """A grid_map layer's value in the cell under each (x, y) point; NaN off the map.

    grid_map puts index (0, 0) at the map's +x, +y corner, top_xy, and counts rows down x and columns down y.
    """
    i = np.floor((top_xy[0] - points[:, 0]) / resolution).astype(int)
    j = np.floor((top_xy[1] - points[:, 1]) / resolution).astype(int)
    inside = (i >= 0) & (i < layer.shape[0]) & (j >= 0) & (j < layer.shape[1])
    out = np.full(len(points), np.nan)
    out[inside] = layer[i[inside], j[inside]]
    return out


def compute_scan(observed, filled, top_xy, resolution, pelvis_xyz, yaw, feet_ground):
    """The policy's scan from an elevation map.

    observed:    the map's measured heights (elevation_mapping_cupy's `elevation` layer), NaN where never seen.
    filled:      its gap-filled heights (`inpaint`), or None.
    feet_ground: the ground height under the feet (the lower sole).

    The upside-down Mid-360 on the head can't see the ground within about 0.94 m (its view stops 52 deg below
    horizontal), and that's most of the scan. The map remembers cells seen from farther away; for the rest:
      - cells the map has observed are used as they are;
      - unobserved points less than NEAR from the pelvis, under the robot, take the ground under the feet;
      - unobserved points farther out take the inpainted layer, or the feet's ground if that has nothing either.
    Filling the whole blind zone from the feet was worse in sim: beside the top of a step it read the drop as flat,
    and the robot walked off the side. Inpainting under the robot too was worse next to walls (inpainting fills
    the hole with the wall's top), where the robot fell standing. Training never saw unknown cells, so these are
    choices, not a match.

    Returns (scan, observed share, from-feet share): the 693 values and the fraction of points from each source.
    The rest of the points are inpainted.
    """
    pelvis_xyz = np.asarray(pelvis_xyz, dtype=float)
    points = scan_points(pelvis_xyz[:2], yaw)
    ground = sample(observed, top_xy, resolution, points)
    fill = sample(filled, top_xy, resolution, points) if filled is not None else np.full(N_POINTS, np.nan)
    seen = np.isfinite(ground)
    from_feet = ~seen & (_NEAR_MASK | ~np.isfinite(fill))
    ground = np.where(seen, ground, np.where(from_feet, feet_ground, fill))
    scan = np.clip(pelvis_xyz[2] - ground, *CLIP)
    return scan, float(seen.mean()), float(from_feet.mean())
