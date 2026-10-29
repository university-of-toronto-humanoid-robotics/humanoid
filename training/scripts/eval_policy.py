"""Measure what a trained G1-Comp policy can do, on flat ground, with numbers.

  1. Crouch: stand still and hold a series of commanded pelvis heights.
  2. The same while the arms move randomly (as if IK had taken over).
  3. Walk forward at 0.5 m/s.

For each it prints the height (or speed) reached and how many robots fell.
From training/ inside the container:

  python scripts/eval_policy.py --checkpoint logs/rsl_rl/g1_comp_rough/<date>/model_<N>.pt
"""

import argparse
from dataclasses import asdict

import _setup  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

HEIGHTS = (0.78, 0.70, 0.62, 0.54, 0.46)
HOLD_S = 4.0  # seconds per commanded height; the last second is measured


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--checkpoint", help="trained model_<N>.pt; leave out to test with zero actions")
  parser.add_argument("--task", default="G1-Comp-Rough")
  parser.add_argument("--num-envs", type=int, default=32)
  args = parser.parse_args()

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.scene.terrain.terrain_type = "plane"  # flat ground: isolate the skills
  cfg.scene.terrain.terrain_generator = None
  cfg.events.pop("randomize_terrain", None)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent_cfg.clip_actions)
  if args.checkpoint:
    runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
  else:  # zero actions: only the default pose (plus the crouch feedforward) is held
    def policy(obs):
      return torch.zeros(env.num_envs, env.unwrapped.action_manager.total_action_dim, device=device)

  base = env.unwrapped
  twist = base.command_manager.get_term("twist")
  height = base.command_manager.get_term("base_height")
  arms = base.action_manager.get_term("joint_pos")
  robot = base.scene["robot"]
  steps_per_s = int(round(1.0 / base.step_dt))
  obs = env.get_observations()

  def command(vx: float, h: float, arms_random: bool) -> None:
    twist.time_left[:] = 1e9  # no resampling during the test
    height.time_left[:] = 1e9
    twist.is_heading_env[:] = False
    twist.is_standing_env[:] = vx == 0.0
    twist.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=device)
    height.goal[:] = h
    arms.cfg.override_prob = 1.0 if arms_random else 0.0
    arms._segment_left[:] = 0.0  # apply the arm setting on the next step

  def run(seconds: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    nonlocal obs
    heights, speeds, falls = [], [], 0
    for k in range(int(seconds * steps_per_s)):
      with torch.no_grad():
        obs, _, dones, extras = env.step(policy(obs))
      falls += int((dones & ~extras["time_outs"].bool()).sum().item())
      if k >= (seconds - 1.0) * steps_per_s:
        heights.append(height.measured_height())
        speeds.append(robot.data.root_link_lin_vel_b[:, 0])
    return torch.stack(heights).mean(0), torch.stack(speeds).mean(0), falls

  for arms_random in (False, True):
    print(f"\nCrouch while standing, arms {'moving randomly' if arms_random else 'policy-controlled'}:")
    print("  commanded  reached (mean +/- std)  falls")
    for h in HEIGHTS:
      command(0.0, h, arms_random)
      reached, _, falls = run(HOLD_S)
      print(f"  {h:6.2f} m   {reached.mean():.3f} +/- {reached.std():.3f} m        {falls}")

  command(0.0, HEIGHTS[0], False)
  run(2.0)
  command(0.5, HEIGHTS[0], False)
  _, speed, falls = run(6.0)
  print(f"\nWalk at 0.50 m/s: reached {speed.mean():.2f} +/- {speed.std():.2f} m/s, {falls} falls")


if __name__ == "__main__":
  main()
