"""The walk sim's side of the terrain scan that the team's height-map policies read: the true scan, straight from
the scene, and a check of the live one against it.

The scan's pattern, and how the elevation map becomes a scan, are in the height_scan package
(ros2_ws/src/perception/height_scan/height_scan/scan.py), shared with the robot.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bridge import g1_model
from height_scan.scan import scan_points, to_scan


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler('ZYX')[0]


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


class ScanMonitor:
    """Compares the scan the controller is being given, /height_scan from height_scan_node, with the true one.

    height_scan_node samples the live elevation map at the pelvis pose that TF gives (FAST-LIO's estimate, with its
    drift and latency), not the true one. See height_scan/scan.py for how it fills the ground the LiDAR hasn't
    seen. Only a report: nothing here reaches the controller.
    """

    TOPIC = '/height_scan'

    def __init__(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import Float32, Float32MultiArray
        self.scan = None
        self.observed = self.from_feet = np.nan
        node.create_subscription(Float32MultiArray, self.TOPIC, self.on_scan, qos_profile_sensor_data)
        node.create_subscription(Float32, self.TOPIC + '/observed_share', self.on_observed, 10)
        node.create_subscription(Float32, self.TOPIC + '/feet_share', self.on_feet, 10)
        self.steps = []   # (observed share, from-feet share, error RMSE over all, max |error|)

    def on_scan(self, msg):
        self.scan = np.array(msg.data, dtype=float)

    def on_observed(self, msg):
        self.observed = msg.data

    def on_feet(self, msg):
        self.from_feet = msg.data

    def record(self, truth):
        """Call once per policy step with the true scan at that moment."""
        if self.scan is None:
            return
        error = self.scan - truth
        self.steps.append((self.observed, self.from_feet, np.sqrt(np.mean(error ** 2)), np.abs(error).max()))

    def report(self):
        if not self.steps:
            return f'Elevation-map scan: no {self.TOPIC} ever arrived.'
        s = np.array(self.steps)
        return (f'Elevation-map scan vs the true one over {len(s)} policy steps: '
                f'{np.nanmean(s[:, 0]) * 100:.0f}% of points observed by the map, '
                f'{np.nanmean(s[:, 1]) * 100:.0f}% from the feet, the rest inpainted; error RMSE '
                f'{s[:, 2].mean() * 100:.1f} cm; worst point per step {np.median(s[:, 3]) * 100:.0f} cm median, '
                f'{np.percentile(s[:, 3], 95) * 100:.0f} cm 95th percentile.')
