"""Record a video of a trained policy without a viewer (for Colab, or to share).

The robot is told to walk forward along +x (the lane direction in lane worlds).
From training/:

  python scripts/record_video.py Open --checkpoint logs/rsl_rl/open/<date>/model_<N>.pt
  python scripts/record_video.py Open --checkpoint <...> --fpv      # what the head camera sees

Saves logs/videos/<task>.mp4 (or <task>_fpv.mp4).
"""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task")
  parser.add_argument("--checkpoint", help="trained model_<N>.pt; leave out for zero actions")
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--speed", type=float, default=0.6, help="forward speed command, m/s")
  parser.add_argument("--fpv", action="store_true", help="record the head camera instead of the outside view")
  parser.add_argument("--out", default="logs/videos")
  args = parser.parse_args()
  if args.fpv:
    os.environ["G1_FPV"] = "1"  # read when our tasks are registered (import _setup)

  import _setup  # noqa: F401
  import imageio.v2 as imageio
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  # Outside view: from behind and above (from the side, lane walls hide the robot).
  cfg.viewer.azimuth, cfg.viewer.elevation, cfg.viewer.distance = 20.0, -35.0, 3.5
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  agent_cfg = load_rl_cfg(args.task)
  base = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None if args.fpv else "rgb_array")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  if args.checkpoint:
    runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
  else:
    def policy(obs):
      return torch.zeros(1, base.action_manager.total_action_dim, device=device)

  twist = base.command_manager.get_term("twist")
  steps = int(args.seconds / base.step_dt)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  path = out / f"{args.task}{'_fpv' if args.fpv else ''}.mp4"
  obs = env.get_observations()
  with imageio.get_writer(path, fps=round(1.0 / base.step_dt)) as video:
    for _ in range(steps):
      # Walk forward, facing +x; hold the command (no resampling).
      twist.time_left[:] = 1e9
      twist.is_standing_env[:] = False
      twist.is_heading_env[:] = True
      twist.heading_target[:] = 0.0
      twist.vel_command_b[:, 0], twist.vel_command_b[:, 1] = args.speed, 0.0
      with torch.no_grad():
        obs, _, _, _ = env.step(policy(obs))
      frame = base.scene["head_camera"].data.rgb[0].cpu().numpy() if args.fpv else base.render()
      video.append_data(frame)
  print(f"saved {path} ({steps} frames)")
  env.close()


if __name__ == "__main__":
  main()
