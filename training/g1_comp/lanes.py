"""Commands and curriculum for the lane worlds (Ramp, Stairs, Open, Competition).

A lane is a corridor along x. With mjlab's uniform heading commands most robots
would be told to walk into a wall, so here most are told to walk along the lane
(either way), with some sideways speed and heading error left in, as a teleoperator
would steer. The curriculum then measures how far each robot got from where it
started, not from the tile's corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LaneVelocityCommandCfg(UniformVelocityCommandCfg):
  lane_prob: float = 0.0
  """Share of heading-controlled robots whose target heading is along the lane
  (+x or -x). 0 gives mjlab's uniform command (used on open ground)."""
  heading_jitter: float = 0.25
  """Lane headings are off by up to this much, in rad (an imperfect operator)."""
  keep_direction_prob: float = 0.7
  """Chance a new command keeps the robot's current lane direction, so most robots
  cross a whole part instead of turning back halfway."""

  def build(self, env: ManagerBasedRlEnv) -> LaneVelocityCommand:
    return LaneVelocityCommand(self, env)


class LaneVelocityCommand(UniformVelocityCommand):
  cfg: LaneVelocityCommandCfg

  def __init__(self, cfg: LaneVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.lane_direction = torch.zeros(self.num_envs, device=self.device)  # 0 (+x) or pi (-x)

  def _random_direction(self, n: int) -> torch.Tensor:
    return torch.where(torch.rand(n, device=self.device) < 0.5, 0.0, math.pi)

  def reset(self, env_ids):
    self.lane_direction[env_ids] = self._random_direction(len(env_ids))
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.cfg.lane_prob <= 0.0:
      return
    n = len(env_ids)
    keep = torch.rand(n, device=self.device) < self.cfg.keep_direction_prob
    direction = torch.where(keep, self.lane_direction[env_ids], self._random_direction(n))
    self.lane_direction[env_ids] = direction
    jitter = (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.heading_jitter
    along = torch.rand(n, device=self.device) < self.cfg.lane_prob
    self.heading_target[env_ids] = torch.where(
      along, wrap_to_pi(direction + jitter), self.heading_target[env_ids]
    )


def mark_start(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Reset event (after the robot is placed): remember where each robot started."""
  # From the joint state just written by the reset: body positions (root_link_pos_w)
  # are only refreshed at the next simulator step and still show the old episode.
  data = env.scene["robot"].data
  pos = data.data.qpos[:, data.indexing.free_joint_q_adr[:2]]
  if not hasattr(env, "_lane_start"):
    env._lane_start = pos.clone()
  if env_ids is None:
    env._lane_start[:] = pos
  else:
    env._lane_start[env_ids] = pos[env_ids]


def lane_terrain_levels(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, promote_distance: float = 3.5
) -> torch.Tensor:
  """Terrain curriculum for lanes: a robot that ended its episode `promote_distance`
  from where it started moves to a harder row; one that covered less than half of
  what its command asked for moves to an easier one (mjlab's rule, from the start
  point instead of the tile origin)."""
  terrain = env.scene.terrain
  assert terrain is not None
  start = getattr(env, "_lane_start", None)
  if start is not None:
    pos = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    distance = torch.norm(pos - start[env_ids], dim=1)
    command = env.command_manager.get_command(command_name)
    move_up = distance > promote_distance
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    terrain.update_env_origins(env_ids, move_up, move_down & ~move_up)
  return torch.mean(terrain.terrain_levels.float())
