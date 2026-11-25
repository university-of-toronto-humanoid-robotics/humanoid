"""Compare student scan conversion with ROS height_scan on the same simulated map.

This checks map-to-scan conversion, not the live ROS elevation mapper. Run from
training/: python scripts/compare_student_ros_scan.py --task descend
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import _setup  # noqa: F401
from g1_comp.rpl import rpl_env_cfg
from g1_comp.student_mapping import attach_student_map
from g1_comp.student_mapping.observation import fill_unknown
from mjlab.envs import ManagerBasedRlEnv


def ros_scan_module():
  path = (Path(__file__).resolve().parents[2] /
          "ros2_ws/src/perception/height_scan/height_scan/scan.py")
  spec = importlib.util.spec_from_file_location("ros_scan_reference", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def ros_projection(env, term, ros):
  """Apply ROS scan.py to a snapshot of the student map and robot pose."""
  h = term.height[:, 0]
  v = term.valid[:, 0]
  filled, fill_valid = fill_unknown(h, v, term.cfg.inpaint_passes)
  # Student tensor is [y ascending, x ascending]. GridMap is [x descending,
  # y descending] with its upper-right corner at center + half the map size.
  def as_grid(tensor):
    return np.flip(tensor[0].detach().cpu().numpy().T, axis=(0, 1)).copy()

  observed = as_grid(h).astype(float)
  observed[~as_grid(v)] = np.nan
  infill = as_grid(filled).astype(float)
  infill[~as_grid(fill_valid)] = np.nan
  names = list(env.scene["robot"].body_names)
  pose = env.scene["robot"].data.body_link_pose_w[0, names.index("pelvis")]
  p = pose[:3].detach().cpu().numpy()
  w, x, y, z = pose[3:7].detach().cpu().numpy()
  yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  feet = env.scene["robot"].data.body_link_pos_w[
    0, [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")], 2
  ]
  foot_ground = float(feet.min().item() - term.cfg.sole_depth)
  top = term.center[0].detach().cpu().numpy() + term.cfg.cells * term.cfg.resolution / 2
  return ros.compute_scan(observed, infill, top, term.cfg.resolution, p, yaw, foot_ground)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=("ascend", "descend", "ramps", "obstacles"), default="descend")
  parser.add_argument("--steps", type=int, default=25)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.steps < 1:
    parser.error("--steps must be positive")
  cfg = attach_student_map(rpl_env_cfg(args.task, play=True))
  cfg.scene.num_envs = 1
  ros = ros_scan_module()
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=args.seed)
    term = env.observation_manager.get_term_cfg("actor", "height_scan").func
    rows = []
    for _ in range(args.steps):
      obs, *_ = env.step(torch.zeros((1, 29), device="cpu"))
      student = obs["actor"][0, -693:].detach().cpu().numpy()
      reference, observed_share, feet_share = ros_projection(env, term, ros)
      diff = student - reference
      rows.append((np.mean(np.abs(diff)), np.sqrt(np.mean(diff * diff)),
                   np.max(np.abs(diff)), observed_share,
                   float((term.last_source == 0).float().mean()), feet_share,
                   float((term.last_source == 1).float().mean())))
    a = np.asarray(rows)
    print(json.dumps({
      "comparison": "ROS scan.py applied to the same simulated student map",
      "task": args.task, "steps": args.steps,
      "scan_mae_m": float(a[:, 0].mean()), "scan_rmse_m": float(a[:, 1].mean()),
      "median_step_max_error_m": float(np.median(a[:, 2])),
      "ros_observed_share": float(a[:, 3].mean()),
      "student_observed_share": float(a[:, 4].mean()),
      "ros_foot_share": float(a[:, 5].mean()),
      "student_foot_share": float(a[:, 6].mean()),
    }, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
