"""Heights, the observation, every reward term and the terminations of HoST.

Reward names carry their group as a prefix (task_, regu_, style_, post_); the
runner (runner.py) adds each group up separately, one critic per group:

  task   the product of its terms (HoST follows "Learning to Get Up"), not x dt
  regu, style, post   sum of weight x value x dt   (host_ground.py:242-279)

Formulas are HoST's (host_ground.py and the terrain variants; paper Tab. VI),
with HoST's own quirks kept on purpose (noted where they differ from the paper).

Heights. `ref` picks how pelvis and head height are measured:
  "host"    HoST's ground/platform/wall: pelvis = world z, head = head z - mean feet z.
  "feet"    HoST's slope: pelvis and head relative to the mean height of the feet.
  "ground"  ours (stairs, mixed worlds): relative to the ground right under the
            pelvis (one downward ray), so lying across stairs can't look like standing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_apply

from . import params as P
from . import robot

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .actions import HostJointAction

GROUND_RAY = "host_ground_ray"
FEET_CONTACT = "host_feet_contact"


# --- Small helpers -----------------------------------------------------------------


def _act(env: ManagerBasedRlEnv) -> HostJointAction:
  return env.action_manager.get_term("joint_pos")  # type: ignore[return-value]


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def _cache(env: ManagerBasedRlEnv) -> dict:
  c = getattr(env, "_host_cache", None)
  if c is None:
    c = {}
    env._host_cache = c
  return c


def _bodies(env: ManagerBasedRlEnv, names: tuple[str, ...]) -> list[int]:
  c = _cache(env)
  key = ("bodies", names)
  if key not in c:
    c[key] = _robot(env).find_bodies(list(names), preserve_order=True)[0]
  return c[key]


def _cols(env: ManagerBasedRlEnv, names: tuple[str, ...]) -> torch.Tensor:
  """Columns of these joints among the policy's joints."""
  c = _cache(env)
  key = ("cols", names)
  if key not in c:
    order = list(robot.policy_joints())
    c[key] = torch.tensor([order.index(n) for n in names], device=env.device, dtype=torch.long)
  return c[key]


def _q(env):
  return _robot(env).data.joint_pos[:, _act(env)._target_ids]


def _qd(env):
  return _robot(env).data.joint_vel[:, _act(env)._target_ids]


def _torque(env):
  return _robot(env).data.actuator_force[:, _act(env).actuator_ids]


def tolerance(x: torch.Tensor, lower: float, upper: float, margin: float, value_at_margin: float) -> torch.Tensor:
  """1 inside [lower, upper]; outside, a Gaussian falling to `value_at_margin` at
  `margin` away (dm_control's tolerance, as HoST g1_utils.py:249-266)."""
  inside = (x >= lower) & (x <= upper)
  if margin == 0:
    return inside.float()
  d = torch.where(x < lower, lower - x, x - upper) / margin
  scale = math.sqrt(-2 * math.log(value_at_margin))
  return torch.where(inside, torch.ones_like(x), torch.exp(-0.5 * (d * scale) ** 2))


# --- Bodies and heights -----------------------------------------------------------

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEES = ("left_knee_link", "right_knee_link")
THIGHS = ("left_hip_pitch_link", "right_hip_pitch_link")


def feet_pos(env) -> torch.Tensor:  # [B, 2, 3]
  return _robot(env).data.body_link_pos_w[:, _bodies(env, FEET)]


def head_pos(env) -> torch.Tensor:  # [B, 3]
  d = _robot(env).data
  i = _bodies(env, ("torso_link",))[0]
  off = torch.tensor(P.HEAD_OFFSET, device=env.device).expand(env.num_envs, 3)
  return d.body_link_pos_w[:, i] + quat_apply(d.body_link_quat_w[:, i], off)


def ground_under_pelvis(env) -> torch.Tensor:  # [B]
  sensor = env.scene[GROUND_RAY]
  hit = sensor.data.hit_pos_w[:, 0, 2]
  missed = sensor.data.distances[:, 0] < 0
  return torch.where(missed, torch.zeros_like(hit), hit)


def reference_height(env, ref: str) -> torch.Tensor:
  if ref == "host":
    return torch.zeros(env.num_envs, device=env.device)
  if ref == "feet":
    return feet_pos(env)[..., 2].mean(dim=1)
  if ref == "ground":
    return ground_under_pelvis(env)
  raise ValueError(f"unknown height reference {ref!r}")


def base_height(env, ref: str = "host") -> torch.Tensor:
  return _robot(env).data.root_link_pos_w[:, 2] - reference_height(env, ref)


def head_height(env, ref: str = "host") -> torch.Tensor:
  if ref == "ground":
    return head_pos(env)[:, 2] - ground_under_pelvis(env)
  return head_pos(env)[:, 2] - feet_pos(env)[..., 2].mean(dim=1)  # HoST, all terrains


class _LiftedRayCfg(GridPatternCfg):
  """One ray straight down, starting 1.5 m above the pelvis."""

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += 1.5
    return offsets, directions


def ground_ray_cfg() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name=GROUND_RAY,
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="world",
    pattern=_LiftedRayCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=5.0,
    include_geom_groups=(0,),  # terrain only (robot shapes are groups 2 and 3)
  )


# --- Observation --------------------------------------------------------------------


def host_proprio(env: ManagerBasedRlEnv, noise: bool = True) -> torch.Tensor:
  """One frame: [pelvis ang vel x0.25, projected gravity, q, qd x0.05, action, beta]
  (76 numbers for 23 joints; host_ground.py:281-297). Noise is added here and the
  frame zeroed during the unactuated phase, both in HoST's order."""
  act = _act(env)
  d = _robot(env).data
  parts = [
    d.root_link_ang_vel_b * P.OBS_SCALE_ANG_VEL,
    d.projected_gravity_b,
    _q(env) * P.OBS_SCALE_DOF_POS,
    _qd(env) * P.OBS_SCALE_DOF_VEL,
    act.applied,
    act.beta,
  ]
  frame = torch.cat(parts, dim=-1)
  if noise:
    c = _cache(env)
    if "noise_vec" not in c:
      nj = act.action_dim
      c["noise_vec"] = torch.cat([
        torch.full((3,), P.NOISE_ANG_VEL * P.OBS_SCALE_ANG_VEL),
        torch.full((3,), P.NOISE_GRAVITY),
        torch.full((nj,), P.NOISE_DOF_POS * P.OBS_SCALE_DOF_POS),
        torch.full((nj,), P.NOISE_DOF_VEL * P.OBS_SCALE_DOF_VEL),
        torch.zeros(nj),
        torch.full((1,), P.NOISE_BETA),
      ]).to(env.device)
    frame = frame + (2 * torch.rand_like(frame) - 1) * c["noise_vec"]
  return frame * act.actuated_after_step.unsqueeze(1)


# --- Task rewards (multiplied together by the runner) ------------------------------


def head_height_reward(env, ref: str = "host") -> torch.Tensor:
  return tolerance(head_height(env, ref), P.HEAD_HEIGHT_TARGET, math.inf, P.HEAD_MARGIN, 0.1)


def orientation_reward(env) -> torch.Tensor:
  return tolerance(-_robot(env).data.projected_gravity_b[:, 2], P.ORIENTATION_BOUND, math.inf, 1.0, 0.05)


# --- Regularization ------------------------------------------------------------------


def dof_acc(env) -> torch.Tensor:
  return torch.sum(torch.square((_act(env).last_joint_vel - _qd(env)) / env.step_dt), dim=1)


def action_rate(env) -> torch.Tensor:
  a = _act(env)
  return torch.sum(torch.square(a.applied - a.last_actions), dim=1)


def action_smoothness(env) -> torch.Tensor:
  a = _act(env)
  return torch.sum(torch.square(a.applied - 2 * a.last_actions + a.last_last_actions), dim=1)


def torques(env) -> torch.Tensor:
  return torch.sum(torch.square(_torque(env)), dim=1)


def joint_power(env) -> torch.Tensor:
  return torch.sum(torch.abs(_torque(env)) * torch.abs(_qd(env)), dim=1)


def dof_vel(env) -> torch.Tensor:
  return torch.sum(torch.square(_qd(env)), dim=1)


def tracking_error(env) -> torch.Tensor:
  return torch.sum(torch.square(_act(env).joint_targets - _q(env)), dim=1)


def _soft_limits(env) -> torch.Tensor:
  c = _cache(env)
  if "soft_limits" not in c:
    lim = _robot(env).data.joint_pos_limits[0, _act(env)._target_ids]  # [J, 2]
    c["soft_limits"] = lim * P.SOFT_DOF_POS_LIMIT  # HoST scales each bound by 0.9 (host_ground.py:395)
  return c["soft_limits"]


def dof_pos_limits(env) -> torch.Tensor:
  q, lim = _q(env), _soft_limits(env)
  return torch.sum(-(q - lim[:, 0]).clip(max=0.0) + (q - lim[:, 1]).clip(min=0.0), dim=1)


def dof_vel_limits(env) -> torch.Tensor:
  c = _cache(env)
  if "vel_limits" not in c:
    c["vel_limits"] = torch.tensor([robot.vel_limit(j) for j in robot.policy_joints()], device=env.device)
  return torch.sum((torch.abs(_qd(env)) - c["vel_limits"] * P.SOFT_DOF_VEL_LIMIT).clip(min=0.0, max=1.0), dim=1)


# --- Style ---------------------------------------------------------------------------


def _stage(env, threshold: float, ref: str) -> torch.Tensor:
  return (base_height(env, ref) > threshold).float()


def waist_deviation(env) -> torch.Tensor:
  return (torch.abs(_q(env)[:, _cols(env, ("waist_yaw_joint",))]) > 1.4).float().squeeze(1)


def _pair_deviation(env, names: tuple[str, str], max_bound: float, min_bound: float) -> torch.Tensor:
  q = torch.abs(_q(env)[:, _cols(env, names)])
  return ((q.max(dim=1).values > max_bound) | (q.min(dim=1).values > min_bound)).float()


def hip_yaw_deviation(env) -> torch.Tensor:
  return _pair_deviation(env, ("left_hip_yaw_joint", "right_hip_yaw_joint"), 1.4, 0.9)


def hip_roll_deviation(env) -> torch.Tensor:
  return _pair_deviation(env, ("left_hip_roll_joint", "right_hip_roll_joint"), 1.4, 0.9)


def hip_pitch_deviation(env) -> torch.Tensor:  # prone only (host_ground_prone.py:1432)
  return _pair_deviation(env, ("left_hip_pitch_joint", "right_hip_pitch_joint"), 0.9, 0.8)


def shoulder_roll_deviation(env, left_below: float = -0.02, right_above: float = 0.02) -> torch.Tensor:
  """Arms crossing in front of the body. Wall: 0 / 0; slope: 0.15 / -0.15 (HoST)."""
  q = _q(env)[:, _cols(env, ("left_shoulder_roll_joint", "right_shoulder_roll_joint"))]
  return ((q[:, 0] < left_below) | (q[:, 1] > right_above)).float()


def foot_displacement(env, side: int, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  """Foot under the pelvis once standing (host_ground.py:1127-1143). side 0 = left."""
  d = _robot(env).data
  foot = feet_pos(env)[:, side]
  mse = torch.sum(torch.square(d.root_link_pos_w[:, :2] - foot[:, :2]), dim=-1).clamp(min=0.3)
  low = (foot[:, 2] - reference_height(env, ref)) < 0.3
  return torch.exp(-2.0 * mse) * low.float() * _stage(env, h2, ref)


def knee_deviation(env) -> torch.Tensor:
  q = _q(env)[:, _cols(env, ("left_knee_joint", "right_knee_joint"))]
  return ((torch.abs(q).max(dim=1).values > 2.85) | (q.min(dim=1).values < -0.06)).float()


def _vertical(upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
  v = upper - lower
  return v[..., 2] / torch.norm(v, dim=-1).clamp(min=1e-6)


def shank_orientation(env, h1: float = P.H_STAGE1, ref: str = "host") -> torch.Tensor:
  d = _robot(env).data
  knees = d.body_link_pos_w[:, _bodies(env, KNEES)]
  up = _vertical(knees, feet_pos(env)).mean(dim=1)
  return tolerance(up, 0.8, math.inf, 1.0, 0.1) * _stage(env, h1, ref)


def thigh_orientation(env) -> torch.Tensor:  # prone, in place of the shank one (host_ground_prone.py:1196)
  d = _robot(env).data
  up = _vertical(d.body_link_pos_w[:, _bodies(env, THIGHS)], d.body_link_pos_w[:, _bodies(env, KNEES)])
  return tolerance(up.min(dim=1).values, 0.8, math.inf, 1.0, 0.1)


def ankle_parallel(env) -> torch.Tensor:
  """Soles parallel to the ground: variance of 5 keypoint heights per foot, x10
  inside (host_ground.py:1169-1179; the paper's < 0.05 is on these scaled heights)."""
  d = _robot(env).data
  ids = _bodies(env, FEET)
  pts = torch.tensor(robot.ANKLE_KEYPOINTS, device=env.device)  # [K, 3]
  pos, quat = d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids]  # [B, 2, ...]
  b, k = env.num_envs, pts.shape[0]
  world = pos.unsqueeze(2) + quat_apply(
    quat.unsqueeze(2).expand(b, 2, k, 4).reshape(-1, 4), pts.expand(b, 2, k, 3).reshape(-1, 3)
  ).reshape(b, 2, k, 3)
  var = torch.var(world[..., 2] * 10, dim=2).mean(dim=1)
  return (var < 0.05).float()


def feet_distance(env) -> torch.Tensor:
  f = feet_pos(env)
  return (torch.norm(f[:, 0] - f[:, 1], dim=-1) > 0.9).float()


def feet_stumble(env) -> torch.Tensor:
  force = env.scene[FEET_CONTACT].data.force  # [B, 2, 3], world frame (netforce)
  return torch.any(torch.norm(force[..., :2], dim=-1) > 3 * torch.abs(force[..., 2]), dim=1).float()


def ang_vel_xy(env, threshold: float = P.H_STAGE1, ref: str = "host") -> torch.Tensor:
  w = _robot(env).data.root_link_ang_vel_b[:, :2]
  return torch.exp(-2.0 * torch.sum(torch.square(w), dim=1)) * _stage(env, threshold, ref)


def lower_body_deviation(env) -> torch.Tensor:
  """Legs mirror each other (host_platform.py:1252): variance across the two legs of
  (-hip roll, hip pitch, -hip yaw, knee), summed."""
  left = _q(env)[:, _cols(env, ("left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", "left_knee_joint"))]
  right = _q(env)[:, _cols(env, ("right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", "right_knee_joint"))]
  left = left * torch.tensor([-1.0, 1.0, -1.0, 1.0], device=env.device)
  var = torch.var(torch.stack([left, right], dim=1), dim=1).sum(dim=-1)
  return torch.exp(-2.0 * var)


# --- Post-task (only once the pelvis is above H2) ---------------------------------


def lin_vel_xy(env, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  v = _robot(env).data.root_link_lin_vel_b[:, :2]
  return torch.exp(-5.0 * torch.sum(torch.square(v), dim=1)) * _stage(env, h2, ref)


def feet_height_var(env, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  z = feet_pos(env)[..., 2] * 10
  return torch.exp(-2.0 * torch.abs(z[:, 0] - z[:, 1]).clamp(min=0.2)) * _stage(env, h2, ref)


def upper_pose(env, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  c = _cache(env)
  if "upper" not in c:
    names = tuple(j for j in robot.policy_joints() if any(k in j for k in ("shoulder", "elbow", "wrist", "waist")))
    c["upper"] = (_cols(env, names), torch.tensor([P.UPPER_TARGET.get(n, 0.0) for n in names], device=env.device))
  cols, target = c["upper"]
  mse = torch.sum(torch.square(_q(env)[:, cols] - target), dim=1)
  return torch.exp(-0.1 * mse) * _stage(env, h2, ref)


def upright(env, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  g = _robot(env).data.projected_gravity_b[:, :2]
  return torch.exp(-5.0 * torch.sum(torch.square(g), dim=1)) * _stage(env, h2, ref)


def base_height_reward(env, target: float = P.BASE_HEIGHT_TARGET, h2: float = P.H_STAGE2, ref: str = "host") -> torch.Tensor:
  h = base_height(env, ref)
  return torch.exp(-20.0 * torch.abs(h - target)) * (h > h2).float()  # paper: squared; code: abs


# --- Per-terrain weights in mixed worlds --------------------------------------------


def per_terrain(env, term, kinds: tuple[str, ...], weights: tuple[float, ...], **kw) -> torch.Tensor:
  """`term` times a weight that depends on the robot's terrain (cfg weight: 1)."""
  from .terrains import kind_per_env

  c = _cache(env)
  key = ("w", kinds, weights)
  if key not in c:
    c[key] = torch.tensor(weights, device=env.device)
  return term(env, **kw) * c[key][kind_per_env(env, kinds)]


# --- Terminations -----------------------------------------------------------------


def blew_up(env) -> torch.Tensor:
  """The simulation exploded (host_ground.py:175-179), only checked once acting."""
  d = _robot(env).data
  bad = (torch.abs(_qd(env)).max(dim=1).values > P.MAX_DOF_VEL) | (torch.norm(d.root_link_lin_vel_b, dim=-1) > P.MAX_BASE_VEL)
  return bad & _act(env).actuated


def pelvis_below(env, height: float) -> torch.Tensor:  # platform: rolled off onto the floor
  return _robot(env).data.root_link_pos_w[:, 2] < height


def head_below(env, height: float) -> torch.Tensor:  # wall: slid down flat
  return head_pos(env)[:, 2] < height


def feet_above(env, height: float) -> torch.Tensor:  # platform, wall: climbing onto it
  return (feet_pos(env)[..., 2].mean(dim=1) > height) & _act(env).actuated


def pelvis_below_feet(env, height: float) -> torch.Tensor:  # slope
  return base_height(env, "feet") < height
