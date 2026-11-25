"""Smoke/throughput check for the opt-in ROS-like student observation.

Run from training/: python scripts/check_student_map.py --task ramps --steps 20
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import _setup  # noqa: F401  (adds Unitree and g1_comp to sys.path)
from g1_comp.rpl import rpl_env_cfg
from g1_comp.student_mapping import attach_student_map
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=("ascend", "descend", "ramps", "obstacles"), default="ramps")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=20)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--probe-resolution", type=float, default=0.10)
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")

  from g1_comp.student_mapping import StudentMapCfg
  cfg = attach_student_map(rpl_env_cfg(args.task, play=args.num_envs == 1),
                           StudentMapCfg(probe_resolution=args.probe_resolution))
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset(seed=args.seed)
    term = env.observation_manager.get_term_cfg("actor", "height_scan").func
    if obs["actor"].shape != (args.num_envs, 791) or obs["critic"].shape != (args.num_envs, 806):
      raise RuntimeError("Student map changed the policy observation layout")
    action = torch.zeros((args.num_envs, 29), device=args.device)
    measured = []
    max_age = 0.0
    refreshed = 0
    t0 = time.monotonic()
    for _ in range(args.steps):
      obs, *_ = env.step(action)
      if not torch.isfinite(obs["actor"]).all():
        raise RuntimeError("Student observation contains a nonfinite value")
      exact = envs_mdp.height_scan(env, "terrain_scan")
      estimated = obs["actor"][:, -693:]
      age = float(term.map_age_s.max().item())
      if age > 1 / term.cfg.map_hz + env.step_dt + 1e-5:
        raise RuntimeError("Student map is older than its 3 Hz refresh interval")
      max_age = max(max_age, age)
      refreshed += int(age <= env.step_dt * 0.5)
      measured.append(torch.stack((
        (estimated - exact).abs().mean(),
        (term.last_source == 0).float().mean(),
        (term.last_source == 1).float().mean(),
        (term.last_source == 2).float().mean(),
        term.map_age_s.mean(),
      )).cpu())
    if args.device.startswith("cuda"):
      torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    metrics = torch.stack(measured).mean(dim=0).tolist()
    # A reset must discard the previous episode's map before making a new scan.
    env.reset()
    if term.map_age_s.max().item() > 1e-5:
      raise RuntimeError("Student map was not refreshed on environment reset")
    report = {
      "task": args.task, "num_envs": args.num_envs, "steps": args.steps,
      "device": args.device, "probe_resolution_m": args.probe_resolution,
      "mean_abs_error_vs_exact_m": metrics[0],
      "observed_share": metrics[1], "foot_fallback_share": metrics[2],
      "inpaint_share": metrics[3], "map_age_s": metrics[4],
      "max_map_age_s": max_age, "refresh_steps": refreshed,
      "elapsed_s": elapsed, "env_steps_per_s": args.num_envs * args.steps / elapsed,
    }
    if args.device.startswith("cuda"):
      report["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
