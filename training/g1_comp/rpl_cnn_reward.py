"""Locomotion-only adaptation of RPL's final stair-expert rewards.

All functions return unweighted values. mjlab applies weight and control-step dt.
The policy's noisy height-map observation is never used for reward geometry.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from . import mdp
from .sensors import FOOT_ENV_CONTACT_NAME, SOLE_SENSOR_NAMES

FEET = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
FOOT_SITES = SceneEntityCfg("robot", site_names=["left_foot", "right_foot"])
KNEES = SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"])
TORSO = SceneEntityCfg("robot", body_names=["torso_link"])
LOWER = SceneEntityCfg("robot", joint_names=r"^(left|right)_(hip|knee|ankle).*|^waist_.*")
LOWER_ACT = SceneEntityCfg("robot", actuator_names=r"^(left|right)_(hip|knee|ankle).*|^waist_.*")
ARMS = SceneEntityCfg("robot", joint_names=r"^(left|right)_(shoulder|elbow|wrist).*")
ARM_ACT = SceneEntityCfg("robot", actuator_names=r"^(left|right)_(shoulder|elbow|wrist).*")
HIPS = SceneEntityCfg("robot", joint_names=r"^(left|right)_hip_(roll|yaw)_joint$")
ANKLES = SceneEntityCfg("robot", joint_names=r"^(left|right)_ankle_roll_joint$")
KNEE_JOINTS = SceneEntityCfg("robot", joint_names=r"^(left|right)_knee_joint$")


def foot_force(env):
  # Sensor is primary foot -> secondary terrain; flip to force acting on foot.
  return -env.scene[FOOT_ENV_CONTACT_NAME].data.force


def contact_mask(env, threshold=1.0):
  return torch.linalg.vector_norm(foot_force(env), dim=-1) > threshold


def command(env):
  return env.command_manager.get_command("twist")


def walking(env):
  return ~env.command_manager.get_term("twist").is_standing_env


def track_axis(env, index: int, angular: bool = False, sigma: float = 0.25):
  actual = env.scene["robot"].data.root_link_ang_vel_b if angular else env.scene["robot"].data.root_link_lin_vel_b
  return torch.exp(-torch.square(command(env)[:, index] - actual[:, index]) / sigma)


def contact_velocity(env, feet: SceneEntityCfg = FEET):
  vel = env.scene["robot"].data.body_link_vel_w[:, feet.body_ids, :3]
  raw = (vel.square().sum(-1) * contact_mask(env)).sum(-1)
  return raw * torch.where(walking(env), 1.0, 10.0)


def foot_orientation(env, feet: SceneEntityCfg = FEET):
  quat = env.scene["robot"].data.body_link_pose_w[:, feet.body_ids, 3:7]
  gravity = torch.zeros_like(quat[..., :3]); gravity[..., 2] = -1.0
  local = quat_apply_inverse(quat, gravity)
  return torch.linalg.vector_norm(local[..., :2], dim=-1).sum(-1)


class AirTime:
  """RPL filtered contact and first-contact timing, once per control step."""

  def __init__(self):
    self.previous = self.timer = self.last = None

  def __call__(self, env):
    current = contact_mask(env)
    if self.timer is None:
      self.previous = torch.zeros_like(current)
      self.timer = torch.zeros_like(current, dtype=torch.float32)
      self.last = torch.zeros_like(self.timer)
    reset = env.episode_length_buf == 1
    self.timer[reset] = 0.0
    self.previous[reset] = False
    filt = current | self.previous
    first = (self.timer > 0) & filt
    self.timer += env.step_dt
    self.last[first] = self.timer[first]
    value = ((self.timer - 0.5) * first).sum(-1)
    value *= (torch.linalg.vector_norm(command(env)[:, :2], dim=-1) > 0.1)
    self.timer *= ~filt
    self.previous = current.clone()
    env._rpl_last_air_time = self.last
    env._rpl_air_time = self.timer
    return value


def air_time_difference(env):
  last = getattr(env, "_rpl_last_air_time", None)
  return torch.zeros(env.num_envs, device=env.device) if last is None else (last[:, 0] - last[:, 1]).abs()


def contact_pattern(env):
  count = contact_mask(env).sum(-1)
  return torch.where(walking(env), count != 1, count != 2).float()


def excess_foot_force(env, max_contact_force: float = 350.0):
  return (torch.linalg.vector_norm(foot_force(env), dim=-1) - max_contact_force).clamp(min=0).sum(-1)


def close_feet(env, sites: SceneEntityCfg = FOOT_SITES, threshold: float = 0.16):
  pos = env.scene["robot"].data.site_pos_w[:, sites.site_ids, :2]
  return (threshold - torch.linalg.vector_norm(pos[:, 0] - pos[:, 1], dim=-1)).clamp(min=0)


def close_knees(env, knees: SceneEntityCfg = KNEES, threshold: float = 0.18):
  pos = env.scene["robot"].data.body_link_pose_w[:, knees.body_ids, :2]
  return (threshold - torch.linalg.vector_norm(pos[:, 0] - pos[:, 1], dim=-1)).clamp(min=0)


def foot_heading_alignment(env, feet: SceneEntityCfg = FEET):
  from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi
  robot = env.scene["robot"].data
  quat = robot.body_link_pose_w[:, feet.body_ids, 3:7]
  unit_x = torch.zeros_like(quat[..., :3]); unit_x[..., 0] = 1.0
  forward = quat_apply(quat, unit_x)
  heading = torch.atan2(forward[..., 1], forward[..., 0])
  base_q = robot.root_link_quat_w
  base_x = torch.zeros_like(base_q[:, :3]); base_x[:, 0] = 1.0
  base_fwd = quat_apply(base_q, base_x)
  base_heading = torch.atan2(base_fwd[:, 1], base_fwd[:, 0])
  return wrap_to_pi(heading - base_heading[:, None]).abs().sum(-1)


def terrain_height(env):
  scan = env.scene["terrain_scan"].data
  z = scan.hit_pos_w[..., 2]
  valid = scan.distances >= 0
  center = z[:, z.shape[1] // 2]
  return center, torch.where(valid, z, center[:, None])


def base_height(env, target: float = 0.78, reduction: float = 0.75, range_m: float = 0.6):
  """Terrain-relative pelvis height; reduce stiffness around height variation."""
  center, all_z = terrain_height(env)
  above = env.scene["robot"].data.root_link_pos_w[:, 2] - center
  variation = (all_z - center[:, None]).abs().amax(-1)
  return (above - target).square() * (1.0 - reduction * (variation / range_m).clamp(0, 1))


def projected_gravity(env):
  quat = env.scene["robot"].data.root_link_quat_w
  gravity = torch.zeros_like(quat[:, :3]); gravity[:, 2] = -1.0
  return quat_apply_inverse(quat, gravity)


def base_orientation(env):
  return projected_gravity(env)[:, :2].square().sum(-1) * walking(env)


def base_roll(env):
  return projected_gravity(env)[:, 1].abs()


def base_pitch(env):
  # RPL factor 1.6 - |commanded torso pitch|; this task fixes that command to zero.
  return projected_gravity(env)[:, 0].abs() * 1.6


def angular_velocity_xy(env, torso: bool = False, torso_cfg: SceneEntityCfg = TORSO):
  data = env.scene["robot"].data
  if torso:
    quat = data.body_link_pose_w[:, torso_cfg.body_ids, 3:7].squeeze(1)
    omega = data.body_link_vel_w[:, torso_cfg.body_ids, 3:6].squeeze(1)
    omega = quat_apply_inverse(quat, omega)
  else:
    omega = data.root_link_ang_vel_b
  return omega[:, :2].square().sum(-1)


def vertical_velocity(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square() * walking(env)


def torque_l2(env, asset_cfg: SceneEntityCfg):
  return env.scene["robot"].data.actuator_force[:, asset_cfg.actuator_ids].square().sum(-1)


def joint_velocity_l2(env, asset_cfg: SceneEntityCfg):
  return env.scene["robot"].data.joint_vel[:, asset_cfg.joint_ids].square().sum(-1)


class ControlStepAcceleration:
  def __init__(self):
    self.previous = None

  def __call__(self, env, asset_cfg: SceneEntityCfg):
    current = env.scene["robot"].data.joint_vel
    if self.previous is None:
      self.previous = current.clone()
    reset = env.episode_length_buf == 1
    self.previous[reset] = current[reset]
    accel = (current[:, asset_cfg.joint_ids] - self.previous[:, asset_cfg.joint_ids]) / env.step_dt
    self.previous = current.clone()
    return accel.square().sum(-1)


def action_rate(env, asset_cfg: SceneEntityCfg):
  ids = asset_cfg.joint_ids  # joint_pos action uses the robot's joint ordering.
  return (env.action_manager.action[:, ids] - env.action_manager.prev_action[:, ids]).square().sum(-1)


def joint_limits(env, asset_cfg: SceneEntityCfg):
  robot = env.scene["robot"].data
  q = robot.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.default_joint_pos_limits[:, asset_cfg.joint_ids]
  return ((limits[..., 0] - q).clamp(min=0) + (q - limits[..., 1]).clamp(min=0)).sum(-1)


def hip_position(env, asset_cfg: SceneEntityCfg = HIPS):
  return env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids].square().sum(-1) * walking(env)


def ankle_roll(env, asset_cfg: SceneEntityCfg = ANKLES):
  return env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids].abs().sum(-1)


def negative_knee(env, asset_cfg: SceneEntityCfg = KNEE_JOINTS, threshold: float = 0.02):
  return (env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids] < threshold).float().sum(-1)


def neutral_arm_pose(env, asset_cfg: SceneEntityCfg = ARMS):
  robot = env.scene["robot"].data
  return (robot.joint_pos[:, asset_cfg.joint_ids] - robot.default_joint_pos[:, asset_cfg.joint_ids]).square().sum(-1)


def failure_termination(env):
  return env.termination_manager.terminated.float()


def stumble(env, ratio: float = 0.5, minimum: float = 5.0):
  return mdp.stumble_from_forces(foot_force(env), ratio, minimum)


def foothold(env):
  return mdp.penalty_foothold(env, FOOT_ENV_CONTACT_NAME, SOLE_SENSOR_NAMES, 0.05, 1.0)


class RawReward:
  """Keep raw and effective coefficients visible without recomputing stateful terms."""

  def __init__(self, name, func, scaled=False):
    self.name, self.func, self.scaled = name, func, scaled

  def __call__(self, env, **params):
    raw = self.func(env, **params)
    if not hasattr(env, "_rpl_raw_rewards"):
      env._rpl_raw_rewards = {}
    env._rpl_raw_rewards[self.name] = raw.detach()
    scale = getattr(env, "_rpl_penalty_scale", 1.0) if self.scaled else 1.0
    return raw * scale


def raw_metric(env, name: str):
  return env._rpl_raw_rewards[name]


def penalty_scale_metric(env):
  scale = getattr(env, "_rpl_penalty_scale", 1.0)
  return torch.ones(env.num_envs, device=env.device) * scale


def forward_error(env):
  return (command(env)[:, 0] - env.scene["robot"].data.root_link_lin_vel_b[:, 0]).abs()


def yaw_error(env):
  return (command(env)[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]).abs()


def sliding_fraction(env, feet: SceneEntityCfg):
  vel = env.scene["robot"].data.body_link_vel_w[:, feet.body_ids, :3]
  contact = contact_mask(env)
  sliding = (torch.linalg.vector_norm(vel[..., :2], dim=-1) > 0.1) & contact
  return sliding.sum(-1).float() / contact.sum(-1).clamp(min=1)


def air_time_last(env, side: int):
  last = getattr(env, "_rpl_last_air_time", None)
  return torch.zeros(env.num_envs, device=env.device) if last is None else last[:, side]


class PenaltyCurriculum:
  """RPL's episode-length update, independent of the terrain curriculum."""

  def __call__(self, env, env_ids, initial_scale=0.1, degree=1e-5,
               down=40, up=210, enabled=True):
    if not hasattr(env, "_rpl_penalty_scale"):
      env._rpl_penalty_scale = torch.tensor(initial_scale if enabled else 1.0, device=env.device)
    lengths = env.episode_length_buf[env_ids]
    if enabled and lengths.numel() and bool((lengths > 0).any()):
      mean = lengths.float().mean()
      multiplier = torch.where(mean < down, 1.0 - degree,
                               torch.where(mean > up, 1.0 + degree, 1.0))
      env._rpl_penalty_scale = (env._rpl_penalty_scale * multiplier).clamp(0, 1)
    return env._rpl_penalty_scale
