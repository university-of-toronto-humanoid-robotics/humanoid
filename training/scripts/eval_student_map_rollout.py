"""Measure the opt-in student scan during a trained expert rollout.

The expert receives the original exact terrain scan. The student observation is
measured alongside it and never controls the robot. This is a simulator test;
it does not run the live ROS elevation mapper.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import _setup  # noqa: F401
from g1_comp.rpl import rpl_env_cfg
from g1_comp.student_mapping import attach_student_map
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls


TASK_IDS = {
  "ascend": "RPL-Ascend-Stairs", "descend": "RPL-Descend-Stairs",
  "ramps": "RPL-Ramps", "obstacles": "RPL-Obstacles",
}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=TASK_IDS, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--terrain-level", type=int, default=9)
  parser.add_argument("--steps", type=int, default=250)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
  if args.steps < 1 or not 0 <= args.terrain_level <= 9:
    parser.error("Use positive steps and terrain level 0–9")

  os.environ["G1_PLAY_TERRAIN_LEVEL"] = str(args.terrain_level)
  os.environ["G1_PLAY_TERRAIN_SEED"] = str(args.seed)
  torch.manual_seed(args.seed)
  cfg = attach_student_map(rpl_env_cfg(args.task, play=True))
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    agent_cfg = load_rl_cfg(TASK_IDS[args.task])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_IDS[args.task])
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    obs = wrapped.get_observations()
    term = env.observation_manager.get_term_cfg("actor", "height_scan").func
    start_x = float(env.scene["robot"].data.root_link_pos_w[0, 0])
    rows = []
    max_progress = 0.0
    with torch.inference_mode():
      for _ in range(args.steps):
        # Keep the teacher on its original observation distribution while the
        # student map is only evaluated, not used to choose actions.
        teacher_obs = obs.clone()
        teacher_actor = teacher_obs["actor"].clone()
        teacher_actor[:, -693:] = envs_mdp.height_scan(env, "terrain_scan").clamp(-1, 2)
        teacher_obs.set("actor", teacher_actor)
        action = policy(teacher_obs)
        obs, _, done, _ = wrapped.step(action)
        student = obs["actor"][:, -693:]
        exact = envs_mdp.height_scan(env, "terrain_scan").clamp(-1, 2)
        error = (student - exact).abs()[0]
        source = term.last_source[0]
        x = float(env.scene["robot"].data.root_link_pos_w[0, 0])
        progress = x - start_x
        max_progress = max(max_progress, progress)
        rows.append((
          float(error.mean()), float(torch.sqrt((error.square()).mean())),
          float(error.max()), float(error[source == 0].mean()) if (source == 0).any() else np.nan,
          float(error[source == 1].mean()) if (source == 1).any() else np.nan,
          float(error[source == 2].mean()) if (source == 2).any() else np.nan,
          float((source == 0).float().mean()), float((source == 1).float().mean()),
          float((source == 2).float().mean()),
          float(exact.max() - exact.min()), float(done[0]), progress,
          float((error > 0.10).float().mean()), float((error > 0.20).float().mean()),
        ))
    a = np.asarray(rows)
    def avg(column):
      return float(np.nanmean(a[:, column]))
    feature = a[(a[:, 11] >= 1.5) & (a[:, 11] <= 3.5)]
    print(json.dumps({
      "task": args.task, "terrain_level": args.terrain_level,
      "checkpoint": str(args.checkpoint), "steps": args.steps,
      "max_forward_progress_m": max_progress,
      "mean_exact_scan_relief_m": avg(9), "peak_exact_scan_relief_m": float(a[:, 9].max()),
      "mean_absolute_error_m": avg(0), "mean_rmse_m": avg(1),
      "p95_step_max_error_m": float(np.percentile(a[:, 2], 95)),
      "observed_error_m": avg(3), "foot_fallback_error_m": avg(4),
      "inpaint_error_m": avg(5), "observed_share": avg(6),
      "foot_fallback_share": avg(7), "inpaint_share": avg(8),
      "episode_end_count": int(a[:, 10].sum()),
      "share_cells_error_gt_10cm": avg(12),
      "share_cells_error_gt_20cm": avg(13),
      "feature_region_steps": len(feature),
      "feature_region_mae_m": float(feature[:, 0].mean()) if len(feature) else None,
      "feature_region_share_gt_20cm": float(feature[:, 13].mean()) if len(feature) else None,
    }, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
