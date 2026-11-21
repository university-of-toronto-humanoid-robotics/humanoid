"""The policy's height map, corrupted the way a real SLAM elevation map is.

On the robot, the map comes from LiDAR SLAM (a world-fixed elevation map,
cropped around the robot's estimated pose). It is never exactly the terrain. In
training, the policy sees the true scan after these corruptions, applied in this
order (the critic keeps the clean scan):

  1. Per-episode shift: one random offset per episode (an episode is one robot's
     life, from spawn to fall or timeout) for the whole map: up to +/-2 cm
     forward/sideways and +/-3 cm up/down. Localization drift and calibration
     error: the map is wrong the same way everywhere.
  2. Per-point noise: every cell, every step, its own error of up to +/-3 cm.
     LiDAR range noise (Livox Mid-360: about 2-3 cm).
  3. Missing points: 2% of the cells, every step, have no data.
  4. Delay: the map is 0-60 ms old (a lag per episode). With a world-fixed SLAM
     map, this is the lag of the robot pose used to crop it.
  5. Blink: rarely, the whole map is missing for 0.1-0.3 s (perception dropout).

Missing cells and blinks read "flat ground at the feet": the robot's pelvis
height above its soles. **The robot-side code must fill holes and dropouts the
same way.**

Sources: Gallant (arXiv 2511.14625, p.4 and p.8), Miki et al. 2022 (arXiv
2201.08117, p.12), CAP (arXiv 2609.11553, p.18). See training/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.envs import mdp as envs_mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

  from .commands import BaseHeightCommand


@dataclass(frozen=True)
class MapNoiseCfg:
  shift_xy: float = 0.02
  """Per-episode horizontal shift of the whole map, +/- m (each axis)."""
  shift_z: float = 0.03
  """Per-episode vertical offset of the whole map, +/- m."""
  point_noise: float = 0.03
  """Per-cell, per-step uniform noise, +/- m."""
  missing_prob: float = 0.02
  """Share of cells with no data, each step."""
  lag_steps: tuple[int, int] = (0, 3)
  """Map age in policy steps (20 ms each), drawn per episode: 0-60 ms."""
  blink_rate: float = 0.001
  """Chance per step that a blink starts (0.001 at 50 Hz: once every 20 s on average)."""
  blink_steps: tuple[int, int] = (5, 15)
  """Blink length in policy steps: 0.1-0.3 s."""


class noisy_height_scan:
  """Observation term: the height scan (pelvis height above each cell, in m)
  with the corruptions above. Same shape and meaning as mjlab's height_scan.

  Params:
    sensor_name: the scan's RayCastSensor.
    grid_shape: (rows, columns) of the scan = (points across, points along).
    resolution: spacing of the scan points, in m.
    noise: a MapNoiseCfg.
    corrupt: False returns the clean scan (play mode, like Unitree's other
      observation noise).
    height_command_name: the crouch command, whose measured pelvis height above
      the soles fills missing cells.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.noise: MapNoiseCfg = p["noise"]
    self.rows, self.cols = p["grid_shape"]
    self.resolution: float = p["resolution"]
    n, dev = env.num_envs, env.device
    self.n, self.device = n, dev
    self.history_len = self.noise.lag_steps[1] + 1
    self.history: torch.Tensor | None = None  # [history_len, envs, cells], made on first call
    self.head = 0
    self.last_step = -1
    self.fresh = torch.ones(n, dtype=torch.bool, device=dev)  # no older maps yet
    self.shift_cells = torch.zeros(n, 2, device=dev)  # (along, across), in cells
    self.offset_z = torch.zeros(n, device=dev)
    self.lag = torch.zeros(n, dtype=torch.long, device=dev)
    self.blink_left = torch.zeros(n, dtype=torch.long, device=dev)
    # Unshifted sample positions, normalized (align_corners=True: -1 and 1 are the edge cells).
    ys = torch.linspace(-1, 1, self.rows, device=dev)
    xs = torch.linspace(-1, 1, self.cols, device=dev)
    self.base_y, self.base_x = torch.meshgrid(ys, xs, indexing="ij")
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
    if isinstance(ids, slice):
      ids = torch.arange(self.n, device=self.device)[ids]
    k, c = len(ids), self.noise

    def uniform(*shape: int, bound: float) -> torch.Tensor:
      return (torch.rand(*shape, device=self.device) * 2 - 1) * bound

    self.shift_cells[ids] = uniform(k, 2, bound=c.shift_xy / self.resolution)
    self.offset_z[ids] = uniform(k, bound=c.shift_z)
    self.lag[ids] = torch.randint(c.lag_steps[0], c.lag_steps[1] + 1, (k,), device=self.device)
    self.blink_left[ids] = 0
    self.fresh[ids] = True

  def _shift(self, scan: torch.Tensor) -> torch.Tensor:
    """Read the map shifted by a fraction of a cell (bilinear, edge cells repeated)."""
    grid = scan.view(self.n, 1, self.rows, self.cols)
    dx = self.shift_cells[:, 0] * 2 / (self.cols - 1)
    dy = self.shift_cells[:, 1] * 2 / (self.rows - 1)
    sample = torch.stack([self.base_x + dx.view(-1, 1, 1), self.base_y + dy.view(-1, 1, 1)], dim=-1)
    out = F.grid_sample(grid, sample, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(self.n, -1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    grid_shape: tuple[int, int],
    resolution: float,
    noise: MapNoiseCfg,
    corrupt: bool = True,
    height_command_name: str = "base_height",
  ) -> torch.Tensor:
    del grid_shape, resolution, noise  # Read in __init__.
    scan = envs_mdp.height_scan(env, sensor_name)
    if not corrupt:
      return scan
    c = self.noise
    term: BaseHeightCommand = env.command_manager.get_term(height_command_name)
    fill = term.measured_height().unsqueeze(1).expand_as(scan)  # flat ground at the feet

    # 1-3: shift, per-point noise, missing cells.
    noisy = self._shift(scan) + self.offset_z.unsqueeze(1)
    noisy = noisy + (torch.rand_like(noisy) * 2 - 1) * c.point_noise
    noisy = torch.where(torch.rand_like(noisy) < c.missing_prob, fill, noisy)

    # 4: delay. Keep the last few maps; serve each robot the one `lag` steps old.
    # The term may be called more than once per step: only a new step advances time.
    if self.history is None:
      self.history = noisy.unsqueeze(0).repeat(self.history_len, 1, 1)
    new_step = env.common_step_counter != self.last_step
    if new_step:
      self.head = (self.head + 1) % self.history_len
      self.last_step = env.common_step_counter
    self.history[self.head] = noisy
    if self.fresh.any():  # just reset: no older maps yet
      self.history[:, self.fresh] = noisy[self.fresh]
      self.fresh[:] = False
    slot = (self.head - self.lag) % self.history_len
    out = self.history[slot, torch.arange(self.n, device=self.device)]

    # 5: blink. The whole map reads "flat at the feet" for a moment.
    if new_step:
      self.blink_left = (self.blink_left - 1).clamp(min=0)
      start = (self.blink_left == 0) & (torch.rand(self.n, device=self.device) < c.blink_rate)
      length = torch.randint(c.blink_steps[0], c.blink_steps[1] + 1, (self.n,), device=self.device)
      self.blink_left = torch.where(start, length, self.blink_left)
    return torch.where((self.blink_left > 0).unsqueeze(1), fill, out)
