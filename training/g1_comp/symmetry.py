"""Mirror augmentation: every training sample is also used mirrored left<->right.

The G1 is left-right symmetric, so a mirrored sample is just as valid as the
original: left and right joints swapped, sideways and turning quantities negated,
the height map flipped across. PPO trains on both, which gives twice the data for
free and an even gait. Mittal et al. 2024 (arXiv 2403.04359, Table I p.6) found
higher returns and much more symmetric behavior; Watch Your Step (p.5) and
Gallant (p.13) use it on the G1.

Every mirror here is a signed permutation of the observation or action vector,
built once per observation group from the term names. An observation term with
no rule raises an error, so adding a new observation can't silently break this:
add its rule to `_term_rule`.

PPO only uses this during its update; acting, saving and the robot are unchanged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms import PPO

from .sensors import SCAN_RESOLUTION, SCAN_SIZE, grid_shape

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from tensordict import TensorDict

_Rule = tuple[list[int], list[float]]  # (source index for each output index, sign)


def joint_mirror(names: list[str]) -> _Rule:
  """Mirror of a joint vector: swap left/right joints; negate roll and yaw joints
  (their axes flip under a left-right reflection, pitch axes don't)."""
  index = {n: i for i, n in enumerate(names)}
  perm, sign = [], []
  for n in names:
    other = n.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
    if other not in index:
      raise KeyError(f"mirror: no partner joint for {n!r}")
    perm.append(index[other])
    sign.append(-1.0 if re.search(r"(roll|yaw)", n) else 1.0)
  return perm, sign


def _swap_feet(per_foot: int, signs: tuple[float, ...] | None = None) -> _Rule:
  """[left..., right...] -> [right..., left...], with optional per-component signs."""
  s = list(signs) if signs is not None else [1.0] * per_foot
  return [*range(per_foot, 2 * per_foot), *range(per_foot)], s + s


def _flip_scan() -> _Rule:
  """Height scan: flip the grid across (rows are the sideways axis)."""
  rows, cols = grid_shape(SCAN_SIZE, SCAN_RESOLUTION)
  perm = [(rows - 1 - r) * cols + c for r in range(rows) for c in range(cols)]
  return perm, [1.0] * (rows * cols)


def _action_names(env: ManagerBasedRlEnv) -> list[str]:
  return list(env.action_manager.get_term("joint_pos")._target_names)


def _term_rule(env: ManagerBasedRlEnv, name: str) -> _Rule:
  joints = list(env.scene["robot"].joint_names)
  rules = {
    # Vectors in the robot frame: (x forward, y left, z up).
    "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),  # roll and yaw rates flip
    "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "command": ([0, 1, 2], [1.0, -1.0, -1.0]),  # (vx, vy, yaw rate)
    # Gait clock: the mirrored robot leads with the other foot, half a period
    # later; sin/cos of a half-period shift is a sign flip (and 0 stays 0).
    "phase": ([0, 1], [-1.0, -1.0]),
    "joint_pos": joint_mirror(joints),
    "joint_vel": joint_mirror(joints),
    "actions": joint_mirror(_action_names(env)),
    "height_scan": _flip_scan(),
    # Per foot, [left, right].
    "foot_height": _swap_feet(1),
    "foot_air_time": _swap_feet(1),
    "foot_contact": _swap_feet(1),
    "foot_contact_forces": _swap_feet(3, (1.0, -1.0, 1.0)),  # yaw frame (mdp.foot_contact_forces_yaw)
    # Symmetric scalars.
    "height_command": ([0], [1.0]),
    "arm_override": ([0], [1.0]),
    "base_height": ([0], [1.0]),
  }
  if name not in rules:
    raise KeyError(f"mirror: no rule for observation term {name!r}; add one in g1_comp/symmetry.py")
  return rules[name]


def term_slices(env: ManagerBasedRlEnv, group: str) -> dict[str, slice]:
  """Where each term sits in a concatenated observation group."""
  manager = env.observation_manager
  out, start = {}, 0
  for name, dims in zip(manager.active_terms[group], manager.group_obs_term_dim[group], strict=True):
    size = 1
    for d in dims:
      size *= int(d)
    out[name] = slice(start, start + size)
    start += size
  return out


def group_mirror(env: ManagerBasedRlEnv, group: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Signed permutation (index, sign) for a whole concatenated observation group."""
  perm: list[int] = []
  sign: list[float] = []
  for name, where in term_slices(env, group).items():
    p, s = _term_rule(env, name)
    size = where.stop - where.start
    if size % len(p):
      raise ValueError(f"mirror: term {name!r} has {size} values, its rule has {len(p)}")
    for block in range(where.start, where.stop, len(p)):  # history frames, if any, are stacked blocks
      perm += [block + i for i in p]
      sign += s
  return torch.tensor(perm, device=env.device), torch.tensor(sign, device=env.device)


def action_mirror(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  p, s = joint_mirror(_action_names(env))
  return torch.tensor(p, device=env.device), torch.tensor(s, device=env.device)


_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _mirror(env: ManagerBasedRlEnv, key: str, x: torch.Tensor) -> torch.Tensor:
  if (id(env), key) not in _cache:
    _cache[(id(env), key)] = action_mirror(env) if key == "__actions__" else group_mirror(env, key)
  perm, sign = _cache[(id(env), key)]
  return x[..., perm] * sign


def mirror_batch(
  env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """RSL-RL data augmentation: [original; mirrored] for observations and actions."""
  base = env.unwrapped
  out_obs = None
  if obs is not None:
    mirrored = obs.clone()
    for key in obs.keys():
      mirrored[key] = _mirror(base, key, obs[key])
    out_obs = torch.cat([obs, mirrored], dim=0)
  out_actions = None
  if actions is not None:
    out_actions = torch.cat([actions, _mirror(base, "__actions__", actions)], dim=0)
  return out_obs, out_actions


class MirrorPPO(PPO):
  """RSL-RL's PPO with mirror augmentation switched on. Used by setting the
  algorithm's class_name, so no new config fields are needed."""

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):  # type: ignore[override]
    cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": "g1_comp.symmetry:mirror_batch",
    }
    alg = PPO.construct_algorithm(obs, env, cfg, device)
    # PPO keeps its own reference. Drop it from the run config, which Unitree's
    # train.py saves to params/agent.yaml afterwards: RSL-RL put the whole
    # environment in it, which can't be written to a file. Put the class name
    # (popped by PPO) back, so agent.yaml records that mirroring was on.
    cfg["algorithm"].pop("symmetry_cfg", None)
    cfg["algorithm"]["class_name"] = "g1_comp.symmetry:MirrorPPO"
    return alg
