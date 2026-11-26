"""HoST's action: a joint-target offset scaled by the action bound, plus the help
the robot gets while learning (the upward pull) and the start of every episode.

Per policy step (50 Hz), for the policy's joints:

    target = q + beta * a          (paper Eq. 1; host_ground.py:424-443)

  - beta is the action bound: 1 at the start of training, lowered to 0.25 by the
    curriculum (curriculum.py), 0.25 in play and evaluation. It is also an input.
  - a comes straight from the policy (tanh mean plus exploration noise, no clip, as HoST).
  - The target is held for the 4 PD substeps (params.TARGET_UPDATE = "policy_step",
    like the real robot) or recomputed from the current q every substep ("substep",
    exactly HoST's code).
  - It reaches the motors 0-20 ms late (params.DELAY_SUBSTEPS, drawn per episode).
  - A constant torque offset of up to 5% of the motor's limit is added per episode
    (HoST's "torque RFI"), through MuJoCo's qfrc_applied.
  - The held joints (waist roll/pitch, wrist pitch/yaw on the 29-DoF robot) get 0.

For the first `unactuated_steps` after a reset the policy is ignored and every
target equals the current position (the robot falls and settles, damped): that is
how HoST gets its lying postures. Observations are zero during that time
(observations.py), and the wall task pushes the robot back against the wall.

The pulling force (paper Sec. IV-C): straight up, `pull_force` newtons (200 at the
start, curriculum.py lowers it), split between torso_link's origin and a point 0.2 m
above it, as HoST's two torso bodies. Only while the policy acts and the trunk is
near vertical (pelvis projected gravity z < -0.8; always for prone, as HoST).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_apply

from . import params as P
from . import robot

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HostJointActionCfg(JointPositionActionCfg):
  unactuated_steps: int = P.UNACTUATED_STEPS
  target_update: str = P.TARGET_UPDATE
  beta_start: float = P.BETA_START
  pull_force_start: float = P.PULL_FORCE_START
  pull: bool = True
  """False: no pulling force at all (play and evaluation)."""
  pull_needs_upright: bool = True
  pushback: float = 0.0
  """Newtons toward -x on the torso during the unactuated phase (the wall task)."""

  def build(self, env: ManagerBasedRlEnv) -> HostJointAction:
    return HostJointAction(self, env)


class HostJointAction(JointPositionAction):
  cfg: HostJointActionCfg

  def __init__(self, cfg: HostJointActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    expected = list(robot.policy_joints())
    if list(self._target_names) != expected:
      raise SystemExit(f"HoST action joints out of order:\n  got      {self._target_names}\n  expected {expected}")
    n, a, dev = self.num_envs, self.action_dim, self.device
    ent = self._entity
    held = list(robot.held_joints())
    self._held_ids = (torch.tensor(ent.find_joints(held, preserve_order=True)[0], device=dev, dtype=torch.long)
                      if held else None)
    act_ids, _ = ent.find_actuators(expected, preserve_order=True)
    self.actuator_ids = torch.tensor(act_ids, device=dev, dtype=torch.long)
    self._v_adr = ent.indexing.joint_v_adr[self._target_ids]
    self._torso = ent.find_bodies("torso_link")[0][0]

    self.effort_limit = torch.tensor([robot.gains(j)[2] for j in expected], device=dev)
    self.beta = torch.full((n, 1), cfg.beta_start, device=dev)
    self.pull_force = torch.full((n, 1), cfg.pull_force_start if cfg.pull else 0.0, device=dev)
    self.steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.actuated = torch.zeros(n, dtype=torch.bool, device=dev)
    # Per robot, set by events.reset_posture from its terrain's recipe.
    self.unactuated = torch.full((n,), cfg.unactuated_steps, dtype=torch.long, device=dev)
    self.pushback = torch.full((n,), cfg.pushback, device=dev)
    # A constant extra force on the torso (N, world frame): eval.py's robustness tests.
    self.extra_force = torch.zeros(n, 3, device=dev)
    # Per-episode randomization, written by events.randomize_actuators.
    self.offset_torque = torch.zeros(n, a, device=dev)
    self.lag = torch.zeros(n, dtype=torch.long, device=dev)
    self._buffer = torch.zeros(P.DELAY_SUBSTEPS[1] + 1, n, a, device=dev)
    self._head = 0
    self._target_now = torch.zeros(n, a, device=dev)
    self._offset_now = torch.zeros(n, a, device=dev)
    self.joint_targets = torch.zeros(n, a, device=dev)
    self.applied = torch.zeros(n, a, device=dev)  # this step's actions, zero while unactuated
    self.last_actions = torch.zeros(n, a, device=dev)
    self.last_last_actions = torch.zeros(n, a, device=dev)
    self.last_joint_vel = torch.zeros(n, a, device=dev)
    self._envs = torch.arange(n, device=dev)
    self._pull_points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, P.PULL_POINT_OFFSET]], device=dev)

  # --- policy step ---------------------------------------------------------------

  def process_actions(self, actions: torch.Tensor) -> None:
    self.last_last_actions[:] = self.last_actions
    self.last_actions[:] = self.applied
    self.last_joint_vel[:] = self._entity.data.joint_vel[:, self._target_ids]
    self._raw_actions[:] = actions
    self.actuated = self.steps > self.unactuated
    a = actions * self.actuated.unsqueeze(1)  # HoST zeroes the actions themselves (host_ground.py:95)
    self.applied[:] = a
    self._offset_now = self.beta * a
    q = self._entity.data.joint_pos[:, self._target_ids]
    self._target_now = q + self._offset_now
    self.steps += 1

  @property
  def actuated_after_step(self) -> torch.Tensor:
    """The observation mask: steps (after this step) > unactuated, as HoST."""
    return self.steps > self.unactuated

  # --- every PD substep --------------------------------------------------------------

  def apply_actions(self) -> None:
    ent = self._entity
    q = ent.data.joint_pos[:, self._target_ids]
    depth = self._buffer.shape[0]
    self._head = (self._head + 1) % depth
    read = (self._head - self.lag) % depth
    if self.cfg.target_update == "substep":
      self._buffer[self._head] = self._offset_now
      target = q + self._buffer[read, self._envs]
    else:
      self._buffer[self._head] = torch.where(self.actuated.unsqueeze(1), self._target_now, q)
      target = self._buffer[read, self._envs]
    target = torch.where(self.actuated.unsqueeze(1), target, q)
    self.joint_targets[:] = target
    ent.set_joint_position_target(target, joint_ids=self._target_ids)
    if self._held_ids is not None:
      ent.set_joint_position_target(torch.zeros(self.num_envs, len(self._held_ids), device=self.device),
                                    joint_ids=self._held_ids)
    ent.data.data.qfrc_applied[:, self._v_adr] = self.offset_torque
    self._apply_wrench()

  def _apply_wrench(self) -> None:
    ent = self._entity
    force = torch.zeros(self.num_envs, 1, 3, device=self.device)
    torque = torch.zeros_like(force)
    on = self.actuated
    if self.cfg.pull_needs_upright:
      on = on & (ent.data.projected_gravity_b[:, 2] < P.FORCE_UPRIGHT_GZ)
    pull = self.pull_force.squeeze(1) * on
    push = -self.pushback * (~self.actuated).float()
    pos = ent.data.body_link_pos_w[:, self._torso]
    quat = ent.data.body_link_quat_w[:, self._torso]
    com = ent.data.body_com_pos_w[:, self._torso]
    half = torch.stack([push / 2, torch.zeros_like(push), pull / 2], dim=-1)  # per application point
    for p in self._pull_points:
      arm = pos + quat_apply(quat, p.expand(self.num_envs, 3)) - com
      torque[:, 0] += torch.cross(arm, half, dim=-1)
    force[:, 0] = 2 * half + self.extra_force * self.actuated.unsqueeze(1)
    ent.write_external_wrench_to_sim(force, torque, body_ids=[self._torso])

  # --- reset ---------------------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self.steps[env_ids] = 0
    self.actuated[env_ids] = False
    self.applied[env_ids] = 0.0
    self.last_actions[env_ids] = 0.0
    self.last_last_actions[env_ids] = 0.0
    self.last_joint_vel[env_ids] = 0.0
    self._buffer[:, env_ids] = 0.0
    self.joint_targets[env_ids] = self._entity.data.joint_pos[env_ids][:, self._target_ids]
