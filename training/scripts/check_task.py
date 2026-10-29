"""Build a task and step it without a policy, to check that it works.

Prints the observation sizes, what the height scan and foot sensors see, how
often robots fall, and the simulation speed. Fast on any GPU. From training/:

  python scripts/check_task.py G1-Comp-Course --num-envs 8 --steps 200
"""

import argparse
import time

import _setup  # noqa: F401
import torch
from g1_comp import mdp as comp_mdp
from g1_comp.sensors import FOOT_SENSOR_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.registry import load_env_cfg


def describe(name: str, x: torch.Tensor) -> str:
  return f"{name}: min {x.min().item():+.3f}  mean {x.mean().item():+.3f}  max {x.max().item():+.3f}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task")
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--play", action="store_true", help="use the task's play config")
  args = parser.parse_args()

  cfg = load_env_cfg(args.task, play=args.play)
  cfg.scene.num_envs = args.num_envs
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  print(f"built {args.task} with {args.num_envs} robots in {time.time() - t0:.1f} s")

  obs, _ = env.reset()
  for group, value in obs.items():
    print(f"observations '{group}': {tuple(value.shape)}")

  scan = envs_mdp.height_scan(env, "terrain_scan")
  misses = (env.scene["terrain_scan"].data.distances < 0).float().mean().item()
  print(f"height scan: {scan.shape[1]} points, {misses * 100:.1f}% of rays missed")
  print("  " + describe("pelvis height above each cell (m)", scan))
  print("  " + describe("foot clearance (m)", comp_mdp.foot_clearance(env, FOOT_SENSOR_NAMES)))
  height = env.command_manager.get_term("base_height")
  print("  " + describe("pelvis height above the soles at reset (m)", height.measured_height()))

  # Zero actions = hold the default pose. Without a policy robots fall eventually.
  actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  arms = env.action_manager.get_term("joint_pos")
  falls, reward, overridden, crouching = 0, 0.0, 0.0, 0.0
  t0 = time.time()
  for _ in range(args.steps):
    obs, rew, terminated, _, _ = env.step(actions)
    falls += int(terminated.sum().item())
    reward += rew.mean().item()
    overridden += arms.overridden.float().mean().item()
    crouching += (height.crouch_depth() > 0.03).float().mean().item()
  dt = time.time() - t0
  print(f"{args.steps} steps in {dt:.1f} s ({args.steps * args.num_envs / dt:,.0f} robot-steps/s); "
        f"{falls} falls; mean reward per step {reward / args.steps:+.3f}")
  print(f"  arms overridden {overridden / args.steps * 100:.0f}% of the time; "
        f"crouch commanded {crouching / args.steps * 100:.0f}% of the time")
  print("  " + describe("commanded pelvis height now (m)", height.height))
  print("  " + describe("foot clearance at the end (m)", comp_mdp.foot_clearance(env, FOOT_SENSOR_NAMES)))
  env.close()


if __name__ == "__main__":
  main()
