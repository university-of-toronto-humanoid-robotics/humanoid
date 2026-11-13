"""The terrain scan that the team's height-map policies read, from the true terrain or the live elevation map.

The pattern is training/g1_comp/sensors.py's terrain_scan: a 33 x 21 grid at 5 cm, 1.6 m along the pelvis heading
by 1.0 m across, centred 0.3 m ahead of the pelvis and turning with its yaw only. Each value is the pelvis height
above the ground at that point, in metres, clipped to [-1, 2]; a point with no ground reads 2. The order is mjlab's
GridPatternCfg: the 21 rows across the heading (right to left) are the outer loop, and the 33 points along it (back
to front) the inner one.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bridge import g1_model

SIZE = (1.6, 1.0)           # m, along the heading x across it
RESOLUTION = 0.05           # m
FORWARD = 0.3               # m, grid centre ahead of the pelvis
CLIP = (-1.0, 2.0)          # m; -1 is "too tall to climb", 2 is "deep drop" or nothing there
STANDING_HEIGHT = 0.78      # m, pelvis above flat ground when standing: the scan of an unknown world, taken as flat

_along = np.arange(-SIZE[0] / 2, SIZE[0] / 2 + RESOLUTION / 2, RESOLUTION)
_across = np.arange(-SIZE[1] / 2, SIZE[1] / 2 + RESOLUTION / 2, RESOLUTION)
_gx, _gy = np.meshgrid(_along, _across, indexing='xy')
OFFSETS = np.stack([_gx.ravel() + FORWARD, _gy.ravel()], axis=1)    # (693, 2) in the pelvis's yaw frame
N_POINTS = len(OFFSETS)


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler('ZYX')[0]


def scan_points(xy, yaw):
    """World (x, y) of each scan point for a pelvis at xy with heading yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return xy + OFFSETS @ np.array([[c, s], [-s, c]])


def to_scan(pelvis_z, ground):
    """Scan values from ground heights (NaN where there is no ground)."""
    return np.clip(np.where(np.isfinite(ground), pelvis_z - ground, CLIP[1]), *CLIP)


class TrueTerrainScan:
    """The scan from the scene itself (geom group 0, never the robot), as training computes it.

    Training casts each ray afresh. Here ground heights are cached on a 1 cm grid, each cell cast once when first
    needed, because casting 693 rays every step is too slow on the competition course. The scene is static, and the
    course's own reference height map is a 1 cm grid too. One difference: rays start high above the scene rather
    than 2 m above the pelvis, so anything taller than that reads as its top (clipped to -1), not as a miss.
    """

    CELL = 0.01   # m
    TOP = 20.0    # m, ray start height

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.cache = {}

    def ground(self, points):
        keys = np.floor(points / self.CELL).astype(np.int64)
        out = np.empty(len(keys))
        for n, key in enumerate(map(tuple, keys)):
            h = self.cache.get(key)
            if h is None:
                cx, cy = (np.array(key) + 0.5) * self.CELL
                g = g1_model.ground_below(self.model, self.data, cx, cy, top=self.TOP)
                h = self.cache[key] = np.nan if g is None else g
            out[n] = h
        return out

    def __call__(self, data):
        pos = data.qpos[:3]
        return to_scan(pos[2], self.ground(scan_points(pos[:2], yaw_of(data.qpos[3:7]))))


class ElevationMapScan:
    """The scan from the live elevation map, as the robot would have it: sampled from the map in odom at the pelvis
    pose that TF gives (FAST-LIO's estimate, with its drift and latency), not the true one.

    The upside-down Mid-360 on the head can't see the ground within about 0.94 m (its view stops 52 deg below
    horizontal), and that's most of the scan. The map remembers cells seen from farther away, but where it never saw
    them (around the start, or beside the path), elevation mapping's inpainting fills them from whatever surrounds the
    hole, which next to a wall is the wall's top. So:
    - cells the map has observed (`elevation` layer) are used as they are;
    - unobserved points within NEAR of the pelvis, under the robot, take the ground under the feet, from the lower
      foot's TF frame (the URDF chain, from joint states) and the sole's depth below it;
    - unobserved points farther out take the `inpaint` layer, or the feet's ground if that has nothing either.
    Filling the whole blind zone from the feet was worse: beside the top of a step it read the drop as flat, and the
    robot walked off the side. Inpainting under the robot too was worse next to walls, where the robot fell standing.
    Before there is a map and a pose, the whole scan is flat ground at standing height. Training never saw unknown
    cells, so these are choices, not a match. It also computes the true scan each step, to report how far the
    policy's input was from it.
    """

    TOPIC = '/elevation_mapping_node/elevation_map_filter'
    NEAR = 0.5       # m: under the robot, where the ground must be at the feet
    FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')

    def __init__(self, node, truth, sole_depth):
        from grid_map_msgs.msg import GridMap
        from tf2_ros import Buffer, TransformListener
        self.truth = truth
        self.sole_depth = sole_depth   # m, sole below the ankle_roll_link origin
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        node.create_subscription(GridMap, self.TOPIC, self.on_map, 1)
        self.map = None
        self.steps = []   # (observed share, from-feet share, error RMSE over observed, error RMSE over all, max |error|)
        self.blind = 0    # steps without a map or pose

    @staticmethod
    def layer(msg, name):
        d = msg.data[list(msg.layers).index(name)]
        cols, rows = d.layout.dim[0].size, d.layout.dim[1].size
        a = np.array(d.data, dtype=float).reshape(cols, rows).T
        return np.roll(a, (-msg.outer_start_index, -msg.inner_start_index), axis=(0, 1))

    def on_map(self, msg):
        observed = self.layer(msg, 'elevation')
        filled = self.layer(msg, 'inpaint') if 'inpaint' in msg.layers else np.full_like(observed, np.nan)
        info = msg.info
        self.map = (observed, filled, info.pose.position.x + info.length_x / 2,
                    info.pose.position.y + info.length_y / 2, info.resolution)

    def lookup(self, frame):
        from rclpy.time import Time
        from tf2_ros import TransformException
        try:
            return self.buffer.lookup_transform('odom', frame, Time()).transform
        except TransformException:
            return None

    def __call__(self, data):
        truth = self.truth(data)
        pelvis = self.lookup('pelvis')
        if self.map is None or pelvis is None:
            self.blind += 1
            return np.full(N_POINTS, STANDING_HEIGHT)
        r = pelvis.rotation
        pos = np.array([pelvis.translation.x, pelvis.translation.y, pelvis.translation.z])
        feet = [t.translation.z - self.sole_depth for t in map(self.lookup, self.FEET) if t is not None]
        feet_ground = min(feet) if feet else pos[2] - STANDING_HEIGHT
        observed, filled, top_x, top_y, res = self.map
        points = scan_points(pos[:2], yaw_of([r.w, r.x, r.y, r.z]))
        i = np.floor((top_x - points[:, 0]) / res).astype(int)   # grid_map: index 0 at the +x, +y corner
        j = np.floor((top_y - points[:, 1]) / res).astype(int)
        inside = (i >= 0) & (i < observed.shape[0]) & (j >= 0) & (j < observed.shape[1])
        ground, fill = np.full(N_POINTS, np.nan), np.full(N_POINTS, np.nan)
        ground[inside], fill[inside] = observed[i[inside], j[inside]], filled[i[inside], j[inside]]
        seen = np.isfinite(ground)
        near = np.hypot(*(points - pos[:2]).T) < self.NEAR
        from_feet = ~seen & (near | ~np.isfinite(fill))
        ground = np.where(seen, ground, np.where(from_feet, feet_ground, fill))
        scan = np.clip(pos[2] - ground, *CLIP)
        error = scan - truth
        self.steps.append((seen.mean(), from_feet.mean(), np.sqrt(np.mean(error[seen] ** 2)) if seen.any() else np.nan,
                           np.sqrt(np.mean(error ** 2)), np.abs(error).max()))
        return scan

    def report(self):
        if not self.steps:
            return 'Elevation-map scan: no map was ever used.'
        s = np.array(self.steps)
        return (f'Elevation-map scan vs the true one over {len(s)} policy steps ({self.blind} more before the first '
                f'map and pose): {s[:, 0].mean() * 100:.0f}% of points observed by the map, {s[:, 1].mean() * 100:.0f}% '
                f'from the feet, the rest inpainted; error RMSE {np.nanmean(s[:, 2]) * 100:.1f} cm on observed points, '
                f'{s[:, 3].mean() * 100:.1f} cm overall; worst point per step {np.median(s[:, 4]) * 100:.0f} cm median, '
                f'{np.percentile(s[:, 4], 95) * 100:.0f} cm 95th percentile.')
