"""Display the current height-map observation passed to a student policy."""

from __future__ import annotations

import numpy as np


class StudentMapPolicy:
  """Capture the policy's current map without changing its action output."""

  def __init__(self, actor):
    self.actor = actor
    self.current_map: np.ndarray | None = None

  def __call__(self, obs):
    values = obs["student"]
    start = self.actor.STATE_DIM
    self.current_map = values[0, start:start + 33 * 21].detach().cpu().numpy().copy()
    return self.actor(obs)


class StudentMapOverlay:
  """Viser overlay shared by the terrain player and competition route viewer."""

  map_color = (30, 210, 255)
  display_lift_m = 0.06

  def setup(self) -> None:
    super().setup()
    self._map_points = self._server.scene.add_point_cloud(
      "/student_height_map", np.zeros((33 * 21, 3), dtype=np.float32),
      colors=self.map_color, point_size=0.025, point_shape="circle", visible=False,
    )
    along = np.linspace(-0.5, 1.1, 33)
    across = np.linspace(-0.5, 0.5, 21)
    x, y = np.meshgrid(along, across, indexing="ij")
    self._map_xy = np.stack((x.ravel(), y.ravel()), axis=-1)
    with self._server.gui.add_folder("Student map"):
      self._map_info = self._server.gui.add_html("Waiting for the first observation")

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    values = self.policy.current_map
    if values is None:
      return
    robot = self.env.unwrapped.scene["robot"]
    pelvis_idx = list(robot.body_names).index("pelvis")
    pose = robot.data.body_link_pose_w[0, pelvis_idx].detach().cpu().numpy()
    w, x, y, z = pose[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    xy = self._map_xy @ np.array([[c, s], [-s, c]]) + pose[:2]
    points = np.column_stack((xy, pose[2] - values + self.display_lift_m))
    self._map_points.points = (points + self._scene._scene_offset).astype(np.float32)
    self._map_points.visible = True
    self._map_info.content = (
      f"Current policy map; height range {np.ptp(values):.2f} m "
      f"(points raised {self.display_lift_m:.2f} m for visibility)"
    )
