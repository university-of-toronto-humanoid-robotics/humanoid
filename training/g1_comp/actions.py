"""Joint position action where the arms are sometimes moved by someone else.

The policy always outputs targets for all 29 joints. For a random share of the
time, though, each robot's arm targets are replaced by a smooth random motion,
as if an IK controller had taken over the arms (pressing a button, carrying a
crate). The legs have to keep the robot balanced through it. `overridden` tells
the policy (as an observation) when its arm outputs are being ignored; on the
robot, set it to 1 whenever the IK drives the arms.

Crouch feedforward: the commanded crouch pose (hip pitch, knee, ankle pitch from
commands.crouch_leg_pitch) is added to the leg targets, so the policy only has to
correct and balance. On the robot, add the same offsets to the policy's targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ARM_JOINTS = (r".*_shoulder_.*", r".*_elbow_.*", r".*_wrist_.*")
# Leg joints the crouch moves, and their column in BaseHeightCommand.leg_pitch_offsets().
CROUCH_JOINTS = ((r".*_hip_pitch_joint", 0), (r".*_knee_joint", 1), (r".*_ankle_pitch_joint", 2))


def crouch_columns(names: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
  """(positions in `names` of the crouch joints, matching offset column)."""
  cols, idx = [], []
  for i, n in enumerate(names):
    for pattern, k in CROUCH_JOINTS:
      if re.fullmatch(pattern, n):
        cols.append(i)
        idx.append(k)
  return torch.tensor(cols, device=device, dtype=torch.long), torch.tensor(idx, device=device, dtype=torch.long)


@dataclass(kw_only=True)
class ArmOverrideJointPositionActionCfg(JointPositionActionCfg):
  arm_joint_names: tuple[str, ...] = ARM_JOINTS
  override_prob: float = 0.3
  """Share of time the arms are overridden."""
  segment_time_range: tuple[float, float] = (2.0, 6.0)
  """How long each overridden / not-overridden stretch lasts, in seconds."""
  goal_time_range: tuple[float, float] = (0.5, 2.0)
  """How often an overridden arm gets a new random goal, in seconds."""
  max_speed: float = 2.0
  """Fastest an overridden arm joint target moves, in rad/s."""
  amplitude: dict[str, float] = field(default_factory=lambda: {
    r".*_shoulder_pitch_.*": 1.0,
    r".*_shoulder_roll_.*": 0.6,
    r".*_shoulder_yaw_.*": 0.8,
    r".*_elbow_.*": 0.8,
    r".*_wrist_.*": 0.6,
  })
  """Random goals are default pose +/- this much (rad), kept inside the soft joint limits."""
  height_command_name: str | None = "base_height"
  """Crouch command whose leg pose is added to the targets; None to turn off."""

  def build(self, env: ManagerBasedRlEnv) -> ArmOverrideJointPositionAction:
    return ArmOverrideJointPositionAction(self, env)


class ArmOverrideJointPositionAction(JointPositionAction):
  cfg: ArmOverrideJointPositionActionCfg

  def __init__(self, cfg: ArmOverrideJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    names = self._target_names
    arm = [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in cfg.arm_joint_names)]
    self._arm_cols = torch.tensor(arm, device=self.device, dtype=torch.long)
    arm_joint_ids = self._target_ids[self._arm_cols]
    self._default = self._entity.data.default_joint_pos[:, arm_joint_ids].clone()
    limits = self._entity.data.soft_joint_pos_limits[:, arm_joint_ids]
    self._lo, self._hi = limits[..., 0], limits[..., 1]
    amp = [next((a for p, a in cfg.amplitude.items() if re.fullmatch(p, names[i])), 0.0) for i in arm]
    self._amp = torch.tensor(amp, device=self.device)
    self._arm_joint_ids = arm_joint_ids
    self._crouch_cols, self._crouch_idx = crouch_columns(names, self.device)

    n = self.num_envs
    self.overridden = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._segment_left = torch.zeros(n, device=self.device)
    self._goal_left = torch.zeros(n, device=self.device)
    self._goal = self._default.clone()
    self._arm_target = self._default.clone()

  def _uniform(self, n: int, lo_hi: tuple[float, float]) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*lo_hi)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    dt = self._env.step_dt

    # Crouch feedforward on the leg pitch joints.
    if self.cfg.height_command_name is not None:
      offsets = self._env.command_manager.get_term(self.cfg.height_command_name).leg_pitch_offsets()
      self._processed_actions[:, self._crouch_cols] += offsets[:, self._crouch_idx]

    # Start or end override stretches. A new override starts from where the arms are.
    self._segment_left -= dt
    switch = (self._segment_left <= 0).nonzero().flatten()
    if len(switch):
      starting = torch.rand(len(switch), device=self.device) < self.cfg.override_prob
      was = self.overridden[switch]
      self.overridden[switch] = starting
      begin = switch[starting & ~was]
      if len(begin):
        self._arm_target[begin] = self._entity.data.joint_pos[begin][:, self._arm_joint_ids]
        self._goal_left[begin] = 0.0
      self._segment_left[switch] = self._uniform(len(switch), self.cfg.segment_time_range)

    # New random goals for overridden arms, approached at a limited speed.
    self._goal_left -= dt
    new = ((self._goal_left <= 0) & self.overridden).nonzero().flatten()
    if len(new):
      noise = torch.rand(len(new), len(self._amp), device=self.device) * 2 - 1
      goal = self._default[new] + noise * self._amp
      self._goal[new] = torch.minimum(torch.maximum(goal, self._lo[new]), self._hi[new])
      self._goal_left[new] = self._uniform(len(new), self.cfg.goal_time_range)
    step = self.cfg.max_speed * dt
    self._arm_target += (self._goal - self._arm_target).clamp(-step, step)

    arm_actions = self._processed_actions[:, self._arm_cols]
    self._processed_actions[:, self._arm_cols] = torch.where(
      self.overridden.unsqueeze(1), self._arm_target, arm_actions
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.overridden[env_ids] = False
    self._segment_left[env_ids] = 0.0  # decide on the first step
    self._arm_target[env_ids] = self._default[env_ids]
    self._goal[env_ids] = self._default[env_ids]
