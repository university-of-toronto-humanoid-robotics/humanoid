"""Crouch command: a target pelvis height above the feet, only while standing.

The policy sees `command` (one number, meters). It equals the standing height
unless the robot is standing still and was asked to crouch, and it moves toward
its target at most `max_rate` m/s, so crouching and standing up are always gradual.
Height is measured from the feet (pelvis z minus mean sole z), so it means the
same thing on the floor, on a stair, or on the landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# G1 flat-footed, upright-torso crouch: pelvis height above the soles (m) and the
# knee angle that gives it (rad). The ankle takes half the knee bend until its
# -0.87 rad limit, the hip the rest (hip + knee + ankle = 0 keeps the foot flat
# under an upright pelvis). Measured on g1.xml with MuJoCo forward kinematics.
CROUCH_HEIGHTS = (0.429, 0.492, 0.563, 0.621, 0.681, 0.728, 0.763, 0.785)
CROUCH_KNEES = (2.20, 2.00, 1.75, 1.50, 1.20, 0.90, 0.60, 0.30)
ANKLE_PITCH_MIN = -0.87


def crouch_leg_pitch(height: torch.Tensor) -> torch.Tensor:
  """(hip pitch, knee, ankle pitch) for a pelvis height, by interpolating the table. [B, 3]"""
  hs = torch.tensor(CROUCH_HEIGHTS, device=height.device)
  ks = torch.tensor(CROUCH_KNEES, device=height.device)
  h = height.clamp(hs[0], hs[-1])
  i = torch.searchsorted(hs, h).clamp(1, len(hs) - 1)
  t = (h - hs[i - 1]) / (hs[i] - hs[i - 1])
  knee = ks[i - 1] + t * (ks[i] - ks[i - 1])
  ankle = torch.clamp(-knee / 2, min=ANKLE_PITCH_MIN)
  return torch.stack([-knee - ankle, knee, ankle], dim=1)


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  foot_site_names: tuple[str, ...] = ("left_foot", "right_foot")
  velocity_command_name: str = "twist"
  standing_threshold: float = 0.1
  """Velocity command (|v_xy| + |w_z|) below which the robot counts as standing."""
  standing_height: float = 0.785
  """Pelvis height above the soles in the G1 home pose, in meters (top of the crouch table)."""
  min_height: float = 0.42
  """Deepest crouch, in meters (deep enough to reach a button at floor level)."""
  crouch_prob: float = 0.7
  """Chance that a robot told to stand still is also told to crouch."""
  max_rate: float = 0.25
  """Fastest the commanded height may change, in m/s."""

  def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
    return BaseHeightCommand(self, env)


class BaseHeightCommand(CommandTerm):
  cfg: BaseHeightCommandCfg

  def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.foot_ids, _ = self.robot.find_sites(cfg.foot_site_names)
    self.goal = torch.full((self.num_envs,), cfg.standing_height, device=self.device)
    self.height = torch.full((self.num_envs,), cfg.standing_height, device=self.device)
    self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["crouch_depth"] = torch.zeros(self.num_envs, device=self.device)
    self._gui_enabled = None  # set by create_gui() in the viser viewer
    self._gui_slider = None
    self._gui_env_idx = None

  @property
  def command(self) -> torch.Tensor:
    return self.height.unsqueeze(1)

  def measured_height(self) -> torch.Tensor:
    """Pelvis height above the mean sole height. [B]"""
    soles = self.robot.data.site_pos_w[:, self.foot_ids, 2].mean(dim=1)
    return self.robot.data.root_link_pos_w[:, 2] - soles

  def crouch_depth(self) -> torch.Tensor:
    """0 = standing height, 1 = deepest crouch. [B]"""
    span = self.cfg.standing_height - self.cfg.min_height
    return ((self.cfg.standing_height - self.height) / span).clamp(0.0, 1.0)

  def leg_pitch_offsets(self) -> torch.Tensor:
    """How far the commanded crouch moves (hip pitch, knee, ankle pitch) from the
    standing pose, in rad. Zero at standing height. [B, 3]

    The joint action adds these to the leg targets (feedforward), and the posture
    rewards measure from the shifted pose. On the robot, add the same offsets.
    """
    standing = crouch_leg_pitch(torch.full_like(self.height, self.cfg.standing_height))
    return crouch_leg_pitch(self.height) - standing

  def is_standing(self) -> torch.Tensor:
    twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    speed = torch.norm(twist[:, :2], dim=1) + torch.abs(twist[:, 2])
    return speed < self.cfg.standing_threshold

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    self.height[env_ids] = self.cfg.standing_height
    return extras

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["height_error"] += torch.abs(self.measured_height() - self.height) / steps
    self.metrics["crouch_depth"] += self.crouch_depth() / steps

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    crouch = torch.rand(n, device=self.device) < self.cfg.crouch_prob
    depth = torch.rand(n, device=self.device)
    target = self.cfg.standing_height - depth * (self.cfg.standing_height - self.cfg.min_height)
    self.goal[env_ids] = torch.where(crouch, target, torch.full_like(target, self.cfg.standing_height))

  def create_gui(self, name, server, get_env_idx) -> None:
    """A height slider in the viser viewer (play.py). The robot only crouches while
    standing still, so also enable Twist with all sliders at 0."""
    with server.gui.add_folder(name.capitalize()):
      self._gui_enabled = server.gui.add_checkbox("Enable", initial_value=False)
      self._gui_slider = server.gui.add_slider(
        "pelvis height (m)", min=self.cfg.min_height, max=self.cfg.standing_height,
        step=0.01, initial_value=self.cfg.standing_height,
      )
    self._gui_env_idx = get_env_idx

  def _update_command(self) -> None:
    if self._gui_enabled is not None and self._gui_enabled.value:
      self.goal[self._gui_env_idx()] = self._gui_slider.value
    target = torch.where(self.is_standing(), self.goal, torch.full_like(self.goal, self.cfg.standing_height))
    step = self.cfg.max_rate * self._env.step_dt
    self.height += (target - self.height).clamp(-step, step)
