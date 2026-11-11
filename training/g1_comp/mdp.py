"""Observation and reward terms for the G1 competition tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from src.tasks.velocity.mdp.rewards import variable_posture

from .actions import ARM_JOINTS, crouch_columns

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.sensor import ContactSensor, RayCastSensor

  from .commands import BaseHeightCommand

_NO_GROUND = 1.0  # m; clearance reported when no ray under a foot hits anything


def foot_clearance(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
  """Height of each foot above the highest ground point under its sole. [B, F]

  Unitree's foot_height uses world z, which is wrong on stairs: a foot resting
  on the third step looks 0.6 m "high".
  """
  out = []
  for name in sensor_names:
    sensor: RayCastSensor = env.scene[name]
    data = sensor.data
    hit_z = torch.where(data.distances >= 0, data.hit_pos_w[..., 2], -torch.inf)
    clearance = data.pos_w[:, 2] - hit_z.max(dim=1).values
    out.append(torch.where(torch.isfinite(clearance), clearance, _NO_GROUND))
  return torch.stack(out, dim=1)


def feet_clearance_terrain(
  env: ManagerBasedRlEnv,
  target_height: float,
  sensor_names: tuple[str, ...],
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize swing-foot height away from `target_height` above the ground below.

  Same as Unitree's feet_clearance (weighted by foot speed, off when standing),
  but measured from the ground under the foot instead of world z = 0.
  """
  asset = env.scene[asset_cfg.name]
  clearance = foot_clearance(env, sensor_names)  # [B, F]
  speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)  # [B, F]
  cost = torch.sum(torch.abs(clearance - target_height) * speed, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      moving = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
      cost = cost * moving.float()
  return cost


# --- Crouch command and arm override -----------------------------------------


def arm_override_flag(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  """1 while the arms are moved by someone else (training: random motion; robot: IK). [B, 1]"""
  return env.action_manager.get_term(action_name).overridden.float().unsqueeze(1)


def measured_base_height(env: ManagerBasedRlEnv, command_name: str = "base_height") -> torch.Tensor:
  """Pelvis height above the soles, in meters (critic only). [B, 1]"""
  term: BaseHeightCommand = env.command_manager.get_term(command_name)
  return term.measured_height().unsqueeze(1)


def base_height_tracking(
  env: ManagerBasedRlEnv, std: float, command_name: str = "base_height"
) -> torch.Tensor:
  """exp(-(error / std)^2) between the measured and commanded pelvis height.

  Only counts while standing or crouched: walking height is left to Unitree's
  rewards, as before (feet on different stairs make the height ambiguous anyway).
  """
  term: BaseHeightCommand = env.command_manager.get_term(command_name)
  error = term.measured_height() - term.height
  active = term.is_standing() | (term.crouch_depth() > 0.03)
  return torch.exp(-torch.square(error / std)) * active.float()


class _CrouchPose:
  """The reward's target pose: the default pose with the commanded crouch added to
  the leg pitch joints (the same offsets the joint action adds), plus a 0/1 mask of
  the arm joints, all in the reward's joint order."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset = env.scene[cfg.params["asset_cfg"].name]
    _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.arm_mask = torch.tensor(
      [any(re.fullmatch(p, n) for p in ARM_JOINTS) for n in names], dtype=torch.float32, device=env.device
    )
    self.crouch_cols, self.crouch_idx = crouch_columns(names, env.device)
    self.crouch_mask = torch.zeros(len(names), device=env.device)
    self.crouch_mask[self.crouch_cols] = 1.0

  def target(self, env, default: torch.Tensor, height_command_name: str) -> torch.Tensor:
    offsets = env.command_manager.get_term(height_command_name).leg_pitch_offsets()
    target = default.clone()
    target[:, self.crouch_cols] += offsets[:, self.crouch_idx]
    return target


def _arm_overridden(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  return env.action_manager.get_term(action_name).overridden.float().unsqueeze(1)


class crouch_aware_posture(variable_posture):
  """Unitree's posture reward, changed so it doesn't fight the new skills:

  - the target pose follows the crouch command (default pose + crouch offsets on
    the hip/knee/ankle pitch joints), with a little extra tolerance (`crouch_std`
    rad at the deepest crouch) on those joints;
  - while the arms are overridden, arm joints don't count (not the policy's doing).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.pose = _CrouchPose(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    height_command_name: str = "base_height",
    action_name: str = "joint_pos",
    crouch_std: float = 0.2,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Resolved in __init__.
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (speed < walking_threshold).float().unsqueeze(1)
    walking = ((speed >= walking_threshold) & (speed < running_threshold)).float().unsqueeze(1)
    running = (speed >= running_threshold).float().unsqueeze(1)
    std = self.std_standing * standing + self.std_walking * walking + self.std_running * running

    depth = env.command_manager.get_term(height_command_name).crouch_depth().unsqueeze(1)
    std = std + crouch_std * depth * self.pose.crouch_mask

    target = self.pose.target(env, self.default_joint_pos[:, asset_cfg.joint_ids], height_command_name)
    error = torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target)
    keep = 1.0 - self.pose.arm_mask * _arm_overridden(env, action_name)
    return torch.exp(-torch.sum(error / std**2 * keep, dim=1) / keep.sum(dim=1))


class crouch_aware_stand_still:
  """Unitree's stand_still penalty (joint offset from the target pose while
  standing), where the target pose follows the crouch command, and the arm joints
  don't count while they are overridden."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.pose = _CrouchPose(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_command_name: str = "base_height",
    action_name: str = "joint_pos",
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = self.pose.target(env, asset.data.default_joint_pos[:, asset_cfg.joint_ids], height_command_name)
    offset = asset.data.joint_pos[:, asset_cfg.joint_ids] - target
    weight = 1.0 - self.pose.arm_mask * _arm_overridden(env, action_name)
    cost = torch.sum(torch.square(offset) * weight, dim=1)
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return cost * (speed <= command_threshold).float()


def skills_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_iteration: int,
  full_iteration: int,
  standing_share: float,
  arm_override_share: float,
  crouch_prob: float,
  steps_per_iteration: int = 24,
  velocity_command_name: str = "twist",
  height_command_name: str = "base_height",
  action_name: str = "joint_pos",
) -> dict[str, torch.Tensor]:
  """Bring in standing, crouching and arm override gradually, after walking.

  Before `start_iteration` the task is Unitree's (5% standing, no crouch, arms
  always the policy's); by `full_iteration` the shares reach their targets.
  Starting with them all on made the young policy learn to fall fast instead.
  """
  del env_ids  # Unused.
  iteration = env.common_step_counter / steps_per_iteration
  progress = min(max((iteration - start_iteration) / max(full_iteration - start_iteration, 1), 0.0), 1.0)
  env.command_manager.get_term(velocity_command_name).cfg.rel_standing_envs = (
    0.05 + progress * (standing_share - 0.05)
  )
  env.command_manager.get_term(height_command_name).cfg.crouch_prob = progress * crouch_prob
  env.action_manager.get_term(action_name).cfg.override_prob = progress * arm_override_share
  return {"skills_progress": torch.tensor(progress)}


class arm_idle:
  """Sum of the policy's squared arm outputs while the arms are overridden. [B]

  Overridden arm outputs do nothing, so nothing else in the reward holds them in
  place: they drifted far from zero in training (run rough2), and the arms jumped
  when the override ended. This pulls them back to the default pose.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    term = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
    self.arm_cols = term._arm_cols

  def __call__(self, env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.raw_action[:, self.arm_cols]), dim=1) * term.overridden.float()


# --- Critic input in the robot's frame (so it can be mirrored) ----------------


def foot_contact_forces_yaw(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Unitree's foot_contact_forces (sign * log(1 + |f|) per component), with each
  foot's force turned into the robot's heading frame instead of the world frame,
  so a left-right mirror is a sign flip of y. Critic only. [B, feet * 3]"""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, feet, 3], world frame (reduce="netforce")
  assert force is not None
  heading = yaw_quat(env.scene["robot"].data.root_link_quat_w)
  heading = heading.unsqueeze(1).expand(-1, force.shape[1], -1)
  local = quat_apply_inverse(heading.reshape(-1, 4), force.reshape(-1, 3)).view_as(force).flatten(1)
  return torch.sign(local) * torch.log1p(torch.abs(local))


# --- Assistive upward force (training only) -----------------------------------


def assist_fraction(env: ManagerBasedRlEnv, max_fraction: float, cutoff_level: float) -> float:
  """Share of the robot's weight carried by the assist force: `max_fraction` when
  the robots' mean terrain level is 0, fading to 0 at `cutoff_level`
  (Watch Your Step, arXiv 2610.29612, p.4). 0 without a terrain curriculum."""
  levels = getattr(env.scene.terrain, "terrain_levels", None) if env.scene.terrain is not None else None
  if levels is None or max_fraction <= 0:
    return 0.0
  mean_level = levels.float().mean().item()
  return max(0.0, min(max_fraction, max_fraction * (1.0 - mean_level / cutoff_level)))


def assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_fraction: float,
  cutoff_level: float,
  body_name: str = "pelvis",
) -> None:
  """Reset event: an upward force on the pelvis, a share of the robot's weight,
  that lasts the whole episode (MuJoCo keeps applied forces until changed). Like
  a harness: young robots survive long enough to learn stepping, and the help is
  gone before the hard terrain. A2CF (arXiv 2506.23125, p.6): on the G1, the top
  terrain level at ~4k iterations instead of ~10k."""
  robot = env.scene["robot"]
  body_ids, _ = robot.find_bodies(body_name)
  if not hasattr(env, "_g1_comp_robot_weight"):
    root = int(robot.indexing.body_ids[body_ids[0]])
    env._g1_comp_robot_weight = float(env.sim.mj_model.body_subtreemass[root]) * 9.81
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  lift = assist_fraction(env, max_fraction, cutoff_level) * env._g1_comp_robot_weight
  forces = torch.zeros(len(env_ids), 1, 3, device=env.device)
  forces[..., 2] = lift
  robot.write_external_wrench_to_sim(forces, torch.zeros_like(forces), env_ids=env_ids, body_ids=body_ids)


def assist_force_log(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, max_fraction: float, cutoff_level: float
) -> dict[str, torch.Tensor]:
  """Curriculum term that only logs the assist share (TensorBoard: Curriculum/assist)."""
  del env_ids  # Unused.
  return {"weight_share": torch.tensor(assist_fraction(env, max_fraction, cutoff_level))}


# --- Unsafe stepping penalty ---------------------------------------------------
# Our stair treads (24 cm) leave ~3.5 cm around the 20.6 cm foot. Two small
# penalties, after "Omnidirectional Humanoid Locomotion on Stairs via Unsafe
# Stepping Penalty" (arXiv 2603.07928), which reports near-100% safe steps in sim.


def foot_on_edge(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  patch_shape: tuple[int, int],
  contact_sensor_name: str,
  edge_height: float,
) -> torch.Tensor:
  """Number of feet standing on a step edge: in contact, with neighbouring ground
  points under the sole more than `edge_height` apart (part of the sole hangs over
  a drop or presses against a rise). 15° ramps give ~1 cm, so they don't count. [B]"""
  contact: ContactSensor = env.scene[contact_sensor_name]
  assert contact.data.found is not None
  on_ground = contact.data.found > 0  # [B, feet], [left, right]
  rows, cols = patch_shape
  edges = []
  for name in sensor_names:
    data = env.scene[name].data
    # A ray that hits nothing reads as a deep drop.
    z = torch.where(data.distances >= 0, data.hit_pos_w[..., 2], data.pos_w[:, 2:3] - 3.0)
    grid = z.view(-1, rows, cols)
    along = (grid[:, :, 1:] - grid[:, :, :-1]).abs().flatten(1).amax(dim=1)
    across = (grid[:, 1:, :] - grid[:, :-1, :]).abs().flatten(1).amax(dim=1)
    edges.append(torch.maximum(along, across) > edge_height)
  return (torch.stack(edges, dim=1) & on_ground).float().sum(dim=1)


def foot_stumble(
  env: ManagerBasedRlEnv, contact_sensor_name: str, ratio: float = 5.0, min_force: float = 5.0
) -> torch.Tensor:
  """Number of feet pushed mostly sideways by the ground, e.g. a toe hitting a
  stair riser: horizontal force over `ratio` times the vertical (legged_gym's
  stumble test). [B]"""
  contact: ContactSensor = env.scene[contact_sensor_name]
  force = contact.data.force
  assert force is not None
  horizontal = torch.norm(force[..., :2], dim=-1)
  hit = (horizontal > ratio * force[..., 2].abs()) & (horizontal > min_force)
  return hit.float().sum(dim=1)
