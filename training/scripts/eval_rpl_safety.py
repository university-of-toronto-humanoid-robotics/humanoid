"""Compare RPL stair checkpoints with identical terrain seeds and commands.

From training/:
  python scripts/eval_rpl_safety.py --baseline models/RPL/ascend/model_2498.pt \
    --candidate models/RPL/ascend/model_after_safety.pt --terrain-level 9
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from g1_comp.rpl import rpl_env_cfg


def run_one(task: str, checkpoint: Path, level: int, seed: int, speed: float,
            max_steps: int, device: str, stumble_ratio: float) -> dict:
  os.environ["G1_PLAY_TERRAIN_LEVEL"] = str(level)
  os.environ["G1_PLAY_TERRAIN_SEED"] = str(seed)
  feature = "ascend" if task == "RPL-Ascend-Stairs" else "descend"
  cfg = rpl_env_cfg(feature, play=True, stumble_ratio=stumble_ratio)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  try:
    runner = load_runner_cls(task)(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    env.reset()
    base = env.unwrapped
    twist = base.command_manager.get_term("twist")
    twist.time_left[:] = 1e9
    twist.is_heading_env[:] = False
    twist.is_standing_env[:] = False
    twist.vel_command_b[:] = torch.tensor([speed, 0.0, 0.0], device=device)
    obs = env.get_observations()
    stumble_steps = unsupported_sum = stalled_steps = speed_sum = 0.0
    success = fall = left_lane = False
    steps = 0
    with torch.no_grad():
      for steps in range(1, max_steps + 1):
        obs, _, dones, _ = env.step(policy(obs))
        metrics = {name: float(value[0]) for name, value in base.metrics_manager.get_active_iterable_terms(0)}
        stumble_steps += metrics["stumble_rate"]
        unsupported_sum += metrics["unsupported_sole_samples"]
        success = bool(base.termination_manager.get_term("rpl_feature_complete")[0])
        fall = bool(base.termination_manager.get_term("fell_over")[0])
        left_lane = bool(base.termination_manager.get_term("lane_bounds")[0])
        actual_speed = float(base.scene["robot"].data.root_link_lin_vel_b[0, 0])
        speed_sum += actual_speed
        stalled_steps += actual_speed < 0.05
        if bool(dones[0]):
          break
    return {
      "checkpoint": str(checkpoint), "seed": seed, "level": level,
      "success": success, "fall": fall, "left_lane": left_lane,
      "steps": steps, "seconds": round(steps * base.step_dt, 3),
      "stumble_steps": int(stumble_steps),
      "stumble_rate": round(stumble_steps / steps, 4),
      "unsupported_samples_per_step": round(unsupported_sum / steps, 3),
      "mean_forward_speed": round(speed_sum / steps, 3),
      "stalled_fraction": round(stalled_steps / steps, 4),
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=("RPL-Ascend-Stairs", "RPL-Descend-Stairs"),
                      default="RPL-Ascend-Stairs")
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path)
  parser.add_argument("--terrain-level", type=int, default=9)
  parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
  parser.add_argument("--speed", type=float, default=0.4)
  parser.add_argument("--stumble-ratio", type=float, default=4.0,
                      help="use 0.5 or 4.0 to compare reference thresholds")
  parser.add_argument("--max-steps", type=int, default=1000)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if not 0 <= args.terrain_level <= 9 or args.speed <= 0 or args.max_steps <= 0 or args.stumble_ratio < 0:
    parser.error("level must be 0–9, speed/max-steps positive, and stumble-ratio nonnegative")
  checkpoints = [args.baseline] + ([args.candidate] if args.candidate else [])
  for checkpoint in checkpoints:
    if not checkpoint.is_file():
      parser.error(f"checkpoint not found: {checkpoint}")

  rows = [run_one(args.task, checkpoint.resolve(), args.terrain_level, seed, args.speed,
                  args.max_steps, args.device, args.stumble_ratio)
          for seed in args.seeds for checkpoint in checkpoints]
  summaries = []
  for checkpoint in checkpoints:
    own = [row for row in rows if row["checkpoint"] == str(checkpoint.resolve())]
    summaries.append({
      "checkpoint": str(checkpoint.resolve()),
      "trials": len(own),
      "success_rate": sum(row["success"] for row in own) / len(own),
      "fall_rate": sum(row["fall"] for row in own) / len(own),
      "mean_stumbles_per_trial": sum(row["stumble_steps"] for row in own) / len(own),
      "mean_forward_speed": sum(row["mean_forward_speed"] for row in own) / len(own),
      "mean_stalled_fraction": sum(row["stalled_fraction"] for row in own) / len(own),
    })
  output = args.output or Path("logs/rpl_safety_eval") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps({"task": args.task, "target_speed": args.speed,
                                "stumble_ratio": args.stumble_ratio, "results": rows,
                                "summaries": summaries}, indent=2) + "\n")
  for row in rows:
    print(f"{Path(row['checkpoint']).name} seed={row['seed']}: success={row['success']} fall={row['fall']} "
          f"stumbles={row['stumble_steps']} unsupported/step={row['unsupported_samples_per_step']} "
          f"speed={row['mean_forward_speed']} m/s stalled={row['stalled_fraction']:.1%}")
  for summary in summaries:
    print(f"{Path(summary['checkpoint']).name}: success={summary['success_rate']:.1%} "
          f"falls={summary['fall_rate']:.1%} stumbles/trial={summary['mean_stumbles_per_trial']:.1f} "
          f"speed={summary['mean_forward_speed']:.2f} m/s "
          f"stalled={summary['mean_stalled_fraction']:.1%}")
  print(f"Saved: {output}")


if __name__ == "__main__":
  main()
