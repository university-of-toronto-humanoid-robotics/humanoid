"""Sim only: compares the localizer's pelvis pose on the course (/localizer/pose) with the true one.

The sim's world is the course frame, so the true pelvis pose is MuJoCo's. Prints the error when the sim ends.
"""
import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from bridge.mujoco_mid360_sim import STAMP_OFFSET

HISTORY = 500   # true poses kept: 10 s at the operator's 50 Hz
MAX_GAP = 0.03  # s between an estimate and the nearest true pose for them to be compared


class LocalizationMonitor:

    def __init__(self, node):
        self.node = node
        self.truth = []      # (t, x, y, yaw)
        self.errors = []     # (t, xy error m, yaw error deg)
        node.create_subscription(PoseStamped, '/localizer/pose', self.on_pose, 10)

    def record(self, t, pos, yaw):
        self.truth.append((t, pos[0], pos[1], yaw))
        del self.truth[:-HISTORY]

    def on_pose(self, msg):
        if not self.truth:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - STAMP_OFFSET
        truth = np.array(self.truth)
        nearest = np.argmin(np.abs(truth[:, 0] - t))
        if abs(truth[nearest, 0] - t) > MAX_GAP:
            return
        _, x, y, yaw = truth[nearest]
        q = msg.pose.orientation
        est_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX')[0]
        turn = (est_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        self.errors.append((t, np.hypot(msg.pose.position.x - x, msg.pose.position.y - y), np.degrees(abs(turn))))

    def report(self):
        if not self.errors:
            return 'No /localizer/pose received (localization off, or it never found the robot).'
        e = np.array(self.errors)
        return (f'Localization vs ground truth, {len(e)} poses: position RMSE {np.sqrt(np.mean(e[:, 1]**2)):.3f} m, '
                f'max {e[:, 1].max():.3f} m, final {e[-1, 1]:.3f} m | heading RMSE {np.sqrt(np.mean(e[:, 2]**2)):.2f} '
                f'deg, max {e[:, 2].max():.2f} deg')
