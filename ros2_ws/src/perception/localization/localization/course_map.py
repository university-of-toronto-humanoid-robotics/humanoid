"""The course as a point cloud with normals, built from its height grid (sim/competition/course_heightmap.npz).

The grid only has the top surface of each 1 cm cell. The LiDAR mostly sees two things there: flat tops (floor,
treads, ramps) and vertical faces (walls, risers, the counter, the door frame). So the map gets both:
  tops   one point per cell, its normal from the slope of the grid. Cells on a steep slope are left out: they are
         the edge of a step, and the faces below cover them.
  faces  wherever two neighbouring cells differ by more than STEP, a column of points between the two heights,
         its normal pointing horizontally toward the lower cell.
Both are then thinned to one point per `voxel`-sized cube.

No ROS in here, so it can be tested on its own.
"""
import numpy as np

STEP = 0.03         # m between neighbouring cells that makes a vertical face
MAX_SLOPE = 0.6     # tan of the steepest slope still treated as a top surface (31 deg; the ramps are 15 deg)


def load_grid(path):
    d = np.load(path)
    return d['heights'].astype(float), float(d['x0']), float(d['y0']), float(d['res'])


def voxel_thin(points, normals, voxel):
    """Keeps the first point in each voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return points[first], normals[first]


def tops(heights, x0, y0, res):
    gx, gy = np.gradient(heights, res)
    ok = np.hypot(gx, gy) < MAX_SLOPE
    i, j = np.nonzero(ok)
    points = np.column_stack([x0 + i * res, y0 + j * res, heights[i, j]])
    normals = np.column_stack([-gx[i, j], -gy[i, j], np.ones(len(i))])
    return points, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def faces(heights, x0, y0, res, spacing):
    """Vertical faces between neighbouring cells, along x (axis 0) and along y (axis 1)."""
    points, normals = [], []
    for axis in (0, 1):
        a = heights[:-1, :] if axis == 0 else heights[:, :-1]
        b = heights[1:, :] if axis == 0 else heights[:, 1:]
        i, j = np.nonzero(np.abs(b - a) > STEP)
        low, high = np.minimum(a[i, j], b[i, j]), np.maximum(a[i, j], b[i, j])
        # The face sits halfway between the two cells and faces the lower one.
        x = x0 + (i + (0.5 if axis == 0 else 0.0)) * res
        y = y0 + (j + (0.5 if axis == 1 else 0.0)) * res
        facing = np.where(b[i, j] > a[i, j], -1.0, 1.0)   # b is higher: the face looks back toward a
        n_steps = np.ceil((high - low) / spacing).astype(int)
        repeat = np.repeat(np.arange(len(i)), n_steps)
        offset = np.arange(len(repeat)) - np.repeat(np.cumsum(n_steps) - n_steps, n_steps)
        z = low[repeat] + (offset + 0.5) * (high - low)[repeat] / n_steps[repeat]
        points.append(np.column_stack([x[repeat], y[repeat], z]))
        n = np.zeros((len(repeat), 3))
        n[:, axis] = facing[repeat]
        normals.append(n)
    return np.concatenate(points), np.concatenate(normals)


def build_map(heights, x0, y0, res, voxel=0.05):
    """Points (N, 3) and unit normals (N, 3) of the course, in the course frame."""
    top_p, top_n = voxel_thin(*tops(heights, x0, y0, res), voxel)
    face_p, face_n = voxel_thin(*faces(heights, x0, y0, res, spacing=voxel / 2), voxel)
    return np.concatenate([top_p, face_p]), np.concatenate([top_n, face_n])


def height_at(heights, x0, y0, res, x, y):
    """The grid's height under each (x, y), or NaN outside it."""
    i = np.round((np.asarray(x) - x0) / res).astype(int)
    j = np.round((np.asarray(y) - y0) / res).astype(int)
    inside = (i >= 0) & (i < heights.shape[0]) & (j >= 0) & (j < heights.shape[1])
    out = np.full(i.shape, np.nan)
    out[inside] = heights[i[inside], j[inside]]
    return out


def ground_offset(heights, x0, y0, res, points, bin_size=0.02):
    """How far to raise points so the floor in them sits on the map's floor.

    For each point, the gap between the map's height under it and its own height. Floor points all give the same
    gap and outnumber everything else, so the most common gap is the answer. points: (N, 3), already placed on
    the course in x and y.
    """
    gap = height_at(heights, x0, y0, res, points[:, 0], points[:, 1]) - points[:, 2]
    gap = gap[np.isfinite(gap)]
    if len(gap) == 0:
        return 0.0
    bins = np.floor(gap / bin_size).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    return float((values[np.argmax(counts)] + 0.5) * bin_size)
