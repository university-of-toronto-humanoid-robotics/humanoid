"""The height map the new tasks feed the policy: terrain height relative to the feet.

Each of the 693 values is (ground height at that point) - (height of the soles),
in meters: 0 on flat ground, +0.2 for a 20 cm step up, negative for a drop.

The older G1-Comp-* tasks use "pelvis height above each point" (0.78 on flat
ground, moving with every bob of the pelvis). Measuring from the feet instead
makes flat ground read exactly 0 whatever the body does, so a model trained on
Flat has learned nothing wrong about the map, and stairs are simply new numbers.

**On the robot:** value = map height at the point - (pelvis height in the map -
pelvis height above the soles from the leg joints). Points with no data read 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp

from .map_noise import MapNoiseCfg, noisy_height_scan

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


def _sole_height(env: ManagerBasedRlEnv, height_command_name: str) -> torch.Tensor:
  """Pelvis height above the soles. [B, 1]"""
  return env.command_manager.get_term(height_command_name).measured_height().unsqueeze(1)


class terrain_heights:
  """The policy's map: relative to the feet, with the SLAM-like errors of
  map_noise.py (shift, per-point noise, missing points, delay, blinks). Missing
  points and blinks read 0, "flat at the feet". Same parameters as noisy_height_scan."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.noisy = noisy_height_scan(cfg, env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.noisy.reset(env_ids)

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
    scan = self.noisy(env, sensor_name, grid_shape, resolution, noise, corrupt, height_command_name)
    return _sole_height(env, height_command_name) - scan


def terrain_heights_clean(
  env: ManagerBasedRlEnv, sensor_name: str, height_command_name: str = "base_height"
) -> torch.Tensor:
  """The critic's map: the same, without errors."""
  return _sole_height(env, height_command_name) - envs_mdp.height_scan(env, sensor_name)
