"""How many robots per GPU? Times real training iterations (50 steps of rollout +
one PPO update) of a HoST task at several robot counts, each in a fresh process.
From training/ inside the container (or in the cluster's smoke job):

  python g1_host/scripts/bench.py HoST-Ground --num-envs 4096 8192 16384

Prints steps/s, seconds per iteration, peak GPU memory, hours for the task's
iterations, and recommends a robot count: the largest one that is at least 1.3x
faster (steps/s) than the previous and stays under 70% of the GPU's memory.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict

import _setup  # noqa: F401


def child(task: str, num_envs: int, iters: int) -> None:
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  from g1_host.runner import HostRunner
  from g1_host.tasks import runner_cfg

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  runner = HostRunner(env, asdict(runner_cfg(task)), device="cuda:0")
  torch.cuda.reset_peak_memory_stats()

  def iteration() -> float:
    t0 = time.time()
    with torch.inference_mode():
      obs = env.get_observations()["actor"]
      for _ in range(runner.num_steps):
        obs_td, _, dones, extras = env.step(runner.alg.act(obs))
        obs = obs_td["actor"]
        runner.alg.process_env_step(runner.group_rewards(), dones, extras.get("time_outs"))
      runner.alg.compute_returns(obs)
    runner.alg.update()
    torch.cuda.synchronize()
    return time.time() - t0

  for _ in range(2):  # warm-up (kernel compilation, CUDA graphs)
    iteration()
  times = [iteration() for _ in range(iters)]
  s = sum(times) / len(times)
  total = torch.cuda.get_device_properties(0).total_memory
  print(json.dumps({"num_envs": num_envs, "s_per_iter": s, "steps_per_s": num_envs * runner.num_steps / s,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu_gb": total / 1e9,
                    "gpu": torch.cuda.get_device_name(0)}))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("task")
  ap.add_argument("--num-envs", type=int, nargs="+", default=[4096, 8192, 16384])
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  args = ap.parse_args()
  if args.child:
    child(args.task, args.num_envs[0], args.iters)
    return

  from g1_host import params as P

  iters_task = P.ITERATIONS.get(args.task, 10000)
  rows = []
  for n in args.num_envs:
    out = subprocess.run([sys.executable, __file__, args.task, "--num-envs", str(n), "--iters", str(args.iters),
                          "--child"], capture_output=True, text=True)
    line = next((ln for ln in reversed(out.stdout.splitlines()) if ln.startswith("{")), None)
    if line is None:
      print(f"{n:6d} robots: failed (out of memory?)\n{out.stderr[-1500:]}")
      break
    rows.append(json.loads(line))
    r = rows[-1]
    print(f"{n:6d} robots: {r['steps_per_s']:9,.0f} steps/s  {r['s_per_iter']:5.2f} s/iter  "
          f"{r['peak_gb']:5.1f} of {r['gpu_gb']:.0f} GB  -> {iters_task * r['s_per_iter'] / 3600:5.1f} h for "
          f"{iters_task} iterations ({r['gpu']})", flush=True)
  best = rows[0] if rows else None
  for prev, cur in zip(rows, rows[1:]):
    if cur["steps_per_s"] >= 1.3 * prev["steps_per_s"] and cur["peak_gb"] < 0.7 * cur["gpu_gb"]:
      best = cur
    else:
      break
  if best:
    print(f"RECOMMENDED NUM_ENVS={best['num_envs']}")


if __name__ == "__main__":
  main()
