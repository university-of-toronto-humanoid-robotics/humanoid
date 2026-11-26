"""Resets, per-episode randomization and HoST's curriculum.

Resets (`reset_posture`): each robot gets a posture and a place, then falls and
settles during the unactuated phase (actions.py), which is how HoST makes its
lying postures. A "recipe" per terrain kind says which postures, yaws and drop
height:

  supine      on the back, head toward -x (HoST's ground, platform, slope)
  prone       face down (HoST's ground_prone)
  side_left / side_right   on a side, rolled +-20 deg (the paper: the prone policy
              also handles side-lying; we train on it in the mixed tasks)
  sit         upright with the legs forward (HoST's wall: then pushed back)
  random      any orientation and a random joint pose, dropped 0.6-1.0 m (ours)

The joints start at the recipe's pose x U(scale) + U(+-0.1), inside 0.9 x the
limits (host_ground.py:458-482).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul, random_orientation

from . import mdp
from . import params as P
from .terrains import kind_per_env

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_S = math.sqrt(0.5)
QUATS = {  # (w, x, y, z), world frame
  "supine": (_S, 0.0, -_S, 0.0),
  "prone": (_S, 0.0, _S, 0.0),
  "sit": (1.0, 0.0, 0.0, 0.0),
}
# Sides: rolled 50-90 deg about the body's long axis, from supine (side_*) or from
# prone (side_*_prone), so they mostly settle into their own family and each
# policy (supine / prone) learns the sides that end up on its side of the switch.
SIDE_ROLL = {"side_left": -70 * P.DEG, "side_right": 70 * P.DEG}
SIDE_ROLL_PRONE = {"side_left_prone": 70 * P.DEG, "side_right_prone": -70 * P.DEG}
SIDE_NOISE = 20 * P.DEG


@dataclass(frozen=True)
class Recipe:
  postures: dict[str, float]
  """Posture -> share."""
  yaws: tuple[float, ...] = (0.0,)
  """Yaws (rad) to pick from at random (+-yaw_noise)."""
  yaw_noise: float = 0.0
  drop: float = 0.5
  """Pelvis height above the spawn point (HoST: 0.5 lying, 0.45 sitting)."""
  jitter: tuple[float, float] = (0.0, 0.0)
  """Uniform +- offset in x and y (m)."""
  pose: dict[str, float] = field(default_factory=lambda: dict(P.RESET_POSE))
  joint_scale: tuple[float, float] = P.INIT_JOINT_SCALE
  unactuated: int = P.UNACTUATED_STEPS
  """Policy steps the robot falls and settles before the policy acts."""
  pushback: float = 0.0
  """Push toward -x (N) while settling: the wall (actions.py)."""


def _quats(names: list[str], device) -> torch.Tensor:
  m = len(names)
  q = torch.zeros(m, 4, device=device)
  supine = torch.tensor(QUATS["supine"], device=device)
  for i, name in enumerate(names):
    if name in QUATS:
      q[i] = torch.tensor(QUATS[name], device=device)
  prone = torch.tensor(QUATS["prone"], device=device)
  for rolls, base in ((SIDE_ROLL, supine), (SIDE_ROLL_PRONE, prone)):
    side = [i for i, nm in enumerate(names) if nm in rolls]
    if side:  # roll about world x = the lying body's long axis
      roll = torch.tensor([rolls[names[i]] for i in side], device=device)
      roll = roll + (torch.rand(len(side), device=device) * 2 - 1) * SIDE_NOISE
      r = quat_from_angle_axis(roll, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(len(side), 3))
      q[side] = quat_mul(r, base.expand(len(side), 4))
  rnd = [i for i, nm in enumerate(names) if nm == "random"]
  if rnd:
    q[rnd] = random_orientation(len(rnd), device)
  unknown = set(names) - set(QUATS) - set(SIDE_ROLL) - set(SIDE_ROLL_PRONE) - {"random"}
  if unknown:
    raise ValueError(f"unknown postures {unknown}")
  return q


def _pose_tensor(env, pose: dict[str, float]) -> torch.Tensor:
  ent = env.scene["robot"]
  out = torch.zeros(len(ent.joint_names), device=env.device)
  for i, j in enumerate(ent.joint_names):
    for pattern, v in pose.items():
      if re.fullmatch(pattern, j):
        out[i] = v
  return out


def reset_posture(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, kinds: tuple[str, ...],
                  recipes: tuple[Recipe, ...]) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  ent = env.scene["robot"]
  n = len(env_ids)
  dev = env.device
  kind = kind_per_env(env, kinds)[env_ids]

  # Where: a spawn point of the robot's tile, or the env origin on a plane.
  terrain = env.scene.terrain
  patches = getattr(terrain, "flat_patches", None) or {}
  if "spawn" in patches:
    spots = patches["spawn"]  # [rows, cols, K, 3]
    k = torch.randint(0, spots.shape[2], (n,), device=dev)
    pos = spots[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], k].clone()
  else:
    pos = env.scene.env_origins[env_ids].clone()

  act = mdp._act(env)
  quat = torch.zeros(n, 4, device=dev)
  joint_pos = torch.zeros(n, len(ent.joint_names), device=dev)
  limits = ent.data.joint_pos_limits[env_ids] * P.SOFT_DOF_POS_LIMIT
  for r_idx, recipe in enumerate(recipes):
    sel = (kind == r_idx).nonzero().flatten()
    m = len(sel)
    if m == 0:
      continue
    names = list(recipe.postures)
    share = torch.tensor([recipe.postures[p] for p in names], device=dev)
    picks = torch.multinomial(share / share.sum(), m, replacement=True).tolist()
    chosen = [names[i] for i in picks]
    q = _quats(chosen, dev)
    yaw = torch.tensor(recipe.yaws, device=dev)[torch.randint(0, len(recipe.yaws), (m,), device=dev)]
    yaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * recipe.yaw_noise
    rz = quat_from_angle_axis(yaw, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(m, 3))
    quat[sel] = quat_mul(rz, q)
    act.unactuated[env_ids[sel]] = recipe.unactuated
    act.pushback[env_ids[sel]] = recipe.pushback
    jx, jy = recipe.jitter
    pos[sel, 0] += (torch.rand(m, device=dev) * 2 - 1) * jx
    pos[sel, 1] += (torch.rand(m, device=dev) * 2 - 1) * jy
    rand = torch.tensor([p == "random" for p in chosen], device=dev)
    drop = torch.where(rand, 0.6 + 0.4 * torch.rand(m, device=dev), torch.full((m,), recipe.drop, device=dev))
    pos[sel, 2] += drop

    base = _pose_tensor(env, recipe.pose).expand(m, -1)
    scale = torch.empty(m, base.shape[1], device=dev).uniform_(*recipe.joint_scale)
    offset = (torch.rand(m, base.shape[1], device=dev) * 2 - 1) * P.INIT_JOINT_OFFSET
    jp = base * scale + offset
    lim = limits[sel]
    lo, hi = lim[..., 0], lim[..., 1]
    wild = lo + (hi - lo) * (0.25 + 0.5 * torch.rand_like(lo))  # random posture: middle half of each range
    jp = torch.where(rand.unsqueeze(1), wild, jp)
    joint_pos[sel] = torch.minimum(torch.maximum(jp, lo), hi)

  ent.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  ent.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
  ent.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_actuators(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None,
                        kp_range: tuple[float, float] = P.KP_SCALE, kd_range: tuple[float, float] = P.KD_SCALE,
                        strength_range: tuple[float, float] = P.MOTOR_STRENGTH,
                        offset: float = P.ACTUATION_OFFSET, lag: tuple[int, int] = P.DELAY_SUBSTEPS) -> None:
  """Per episode, for the policy's joints: Kp x U(kp) x s, Kd x U(kd) x s with one
  motor strength s per joint (strength scales the whole PD torque, so it scales both
  gains); a constant torque offset of U(+-offset) x limit; a command delay."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  act = mdp._act(env)
  ent = env.scene["robot"]
  n, j = len(env_ids), act.action_dim
  dev = env.device
  ctrl = ent.indexing.ctrl_ids[act.actuator_ids]
  s = torch.empty(n, j, device=dev).uniform_(*strength_range)
  kp = torch.empty(n, j, device=dev).uniform_(*kp_range) * s
  kd = torch.empty(n, j, device=dev).uniform_(*kd_range) * s
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  rows = env_ids[:, None]
  env.sim.model.actuator_gainprm[rows, ctrl, 0] = gain[ctrl, 0] * kp
  env.sim.model.actuator_biasprm[rows, ctrl, 1] = bias[ctrl, 1] * kp
  env.sim.model.actuator_biasprm[rows, ctrl, 2] = bias[ctrl, 2] * kd
  act.offset_torque[env_ids] = (torch.rand(n, j, device=dev) * 2 - 1) * offset * act.effort_limit
  act.lag[env_ids] = torch.randint(lag[0], lag[1] + 1, (n,), device=dev)


# --- Curriculum ----------------------------------------------------------------------


def force_and_bound(env: ManagerBasedRlEnv, env_ids, head_ref: str = "host", h2: float = P.H_STAGE2,
                    base_ref: str = "host") -> dict[str, torch.Tensor]:
  """HoST's curriculum (host_ground.py:505-508), called with each batch of robots
  whose episode just ended, before they are reset: if their mean head height is
  above 0.9 m, their pull force drops 20 N (to 0) and their action bound 0.02 (to
  0.25). Logged under Curriculum/force_and_bound/."""
  act = mdp._act(env)
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  if len(env_ids) == 0:
    zero = torch.tensor(0.0, device=env.device)
    return {"pull_force": act.pull_force.mean(), "beta": act.beta.mean(), "head_height_end": zero, "success_rate": zero}
  head = mdp.head_height(env, head_ref)[env_ids]
  # The very first reset happens before any step, with the robots still in the
  # model's standing pose: that is not a robot that stood up.
  first_reset = env.common_step_counter == 0
  if not first_reset and float(head.mean()) > P.CURRICULUM_HEAD_HEIGHT:
    act.pull_force[env_ids] = (act.pull_force[env_ids] - P.PULL_FORCE_STEP).clamp(min=0.0)
    act.beta[env_ids] = (act.beta[env_ids] - P.BETA_STEP).clamp(min=P.BETA_MIN)
  stood = (head > P.CURRICULUM_HEAD_HEIGHT) & (mdp.base_height(env, base_ref)[env_ids] > h2)
  return {
    "pull_force": act.pull_force.mean(),
    "beta": act.beta.mean(),
    "head_height_end": head.mean(),
    "success_rate": stood.float().mean(),
  }
