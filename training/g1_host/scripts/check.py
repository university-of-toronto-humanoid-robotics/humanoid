"""Check that a HoST task is wired the way HoST is: build it, drop the robots, and
test the pieces one by one (no trained model needed). From training/:

  python g1_host/scripts/check.py HoST-Ground
  python g1_host/scripts/check.py all --num-envs 32

Checks: observation/action sizes; robot mass and pull-force share; postures after
settling; the pull force only acts when upright and after the settling phase; the
curriculum steps only for robots that stood up; rewards for a standing vs a lying
robot; no NaN under random actions; the held waist joints stay put. Exits with 1
if anything failed.
"""

import argparse
import sys
from dataclasses import asdict

import _setup  # noqa: F401
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from g1_host import mdp
from g1_host import params as P
from g1_host.events import force_and_bound
from g1_host.runner import HostRunner
from g1_host.tasks import SPECS, runner_cfg

failures: list[str] = []


def check(task: str, name: str, ok: bool, detail: str = "") -> None:
  print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
  if not ok:
    failures.append(f"{task}: {name}")


def stand(env, ids) -> None:
  """Put robots upright, at rest, pelvis 0.80 m above their env origin, joints at 0."""
  ent = env.scene["robot"]
  pos = env.scene.env_origins[ids].clone()
  pos[:, 2] += 0.80
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(len(ids), 4)
  ent.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
  ent.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
  q = torch.zeros(len(ids), len(ent.joint_names), device=env.device)
  ent.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
  env.sim.forward()


def run(task: str, num_envs: int) -> None:
  print(f"=== {task}", flush=True)
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  wrapped = RslRlVecEnvWrapper(env)  # resets
  obs = wrapped.get_observations()
  ent = env.scene["robot"]
  act = mdp._act(env)
  n = env.num_envs
  zeros = torch.zeros(n, act.action_dim, device=dev)
  check(task, "observation 6 x 76 = 456, 23 actions", tuple(obs["actor"].shape) == (n, 456) and act.action_dim == 23,
        f"{tuple(obs['actor'].shape)}, {act.action_dim}")
  mass = float(env.sim.model.body_subtreemass[0, ent.indexing.body_ids[0]])
  share = P.PULL_FORCE_START / (mass * 9.81)
  check(task, "pull force ~60% of the weight (HoST)", 0.45 < share < 0.8, f"mass {mass:.1f} kg, {share:.0%}")

  # Settle (unactuated): observations are zero, then the posture is what the task asks for.
  first = int(act.unactuated.min())
  early_nonzero = False
  for i in range(int(act.unactuated.max()) + 5):
    out, *_ = env.step(zeros)
    if i < first - 1 and bool((out["actor"] != 0).any()):
      early_nonzero = True
  check(task, "observations are zero while settling", not early_nonzero)
  g = ent.data.projected_gravity_b
  gx, gz = g[:, 0], g[:, 2]
  spec = SPECS[task]
  if task in ("HoST-Ground", "HoST-Platform", "HoST-Slope"):
    frac = float((gx < -0.7).float().mean())
    check(task, "settled on the back (gravity x < -0.7)", frac > 0.9, f"{frac:.0%}")
  elif task == "HoST-Prone":
    frac = float((gx > 0.7).float().mean())
    check(task, "settled face down (gravity x > 0.7)", frac > 0.9, f"{frac:.0%}")
  elif task == "HoST-Wall":
    frac = float(((gz < -0.3) & (gx.abs() < 0.8)).float().mean())
    check(task, "sitting against the wall, not lying", frac > 0.8, f"{frac:.0%}")
  print(f"  info  after settling: on back {float((gx < -0.5).float().mean()):.0%}, face down "
        f"{float((gx > 0.5).float().mean()):.0%}, upright {float((gz < -0.8).float().mean()):.0%}, "
        f"pelvis {float(mdp.base_height(env, spec.ref).mean()):.2f} m", flush=True)

  # Pull force: off while lying (when the task needs the trunk upright); 200 N up when upright.
  torso = ent.find_bodies("torso_link")[0][0]
  env.step(zeros)
  if spec.pull_needs_upright:
    lying = (ent.data.projected_gravity_b[:, 2] > P.FORCE_UPRIGHT_GZ) & act.actuated
    f_lying = ent.data.body_external_force[:, torso, 2][lying]
    check(task, "no pull force while lying", len(f_lying) == 0 or bool((f_lying.abs() < 1e-6).all()))
  ids = torch.arange(min(4, n), device=dev)
  stand(env, ids)
  act.apply_actions()
  fz = ent.data.body_external_force[ids, torso, 2]
  check(task, "pull force = 200 N up when upright", torch.allclose(fz, act.pull_force[ids, 0], atol=1e-3),
        f"{[round(x, 1) for x in fz.tolist()]}")

  # Curriculum: standing robots step down, the others don't.
  before_f, before_b = act.pull_force.clone(), act.beta.clone()
  env.common_step_counter = max(env.common_step_counter, 1)
  force_and_bound(env, ids)
  others = torch.arange(len(ids), n, device=dev)
  check(task, "curriculum: -20 N and -0.02 bound for robots that stood up",
        torch.allclose(act.pull_force[ids], before_f[ids] - 20) and torch.allclose(act.beta[ids], before_b[ids] - 0.02))
  check(task, "curriculum: others unchanged", torch.equal(act.pull_force[others], before_f[others]))
  act.pull_force[:], act.beta[:] = before_f, before_b

  # Rewards, grouped as the runner does: standing vs lying.
  runner = HostRunner(wrapped, asdict(runner_cfg(task)), device=dev)
  env.reward_manager.compute(dt=env.step_dt)
  grp = runner.group_rewards()
  check(task, "standing: task reward ~1", float(grp[ids, 0].min()) > 0.9, f"{[round(x, 3) for x in grp[ids, 0].tolist()]}")
  check(task, "standing: post-task rewards on", float(grp[ids, 3].min()) > 0.0, f"{float(grp[ids, 3].mean()):.3f}")
  if len(others):
    # Prone: HoST's post-task lower-body symmetry term has no standing gate (host_ground_prone.py:1447).
    names = env.reward_manager.active_terms
    gated = [i for i, nm in enumerate(names) if nm.startswith("post_") and nm != "post_lower_body_deviation"]
    lying_post = float(env.reward_manager._step_reward[others][:, gated].abs().max()) if gated else 0.0
    check(task, "lying: standing-gated post-task rewards off", lying_post < 1e-6, f"max {lying_post:.3g}")

  # Held joints (waist roll/pitch) while lying still, then random actions: no NaN.
  wrapped.reset()
  waist = ent.find_joints(["waist_roll_joint", "waist_pitch_joint"], preserve_order=True)[0]
  for _ in range(int(act.unactuated.max()) + 20):
    env.step(zeros)
  still = float(ent.data.joint_pos[:, waist].abs().max())
  check(task, "held waist roll/pitch stay within 0.2 rad at rest", still < 0.2, f"max {still:.3f} rad")
  worst, finite = 0.0, True
  for _ in range(150):
    out, *_ = env.step(torch.randn(n, act.action_dim, device=dev) * 0.5)
    finite &= bool(torch.isfinite(out["actor"]).all())
    worst = max(worst, float(ent.data.joint_pos[:, waist].abs().max()))
  check(task, "no NaN under random actions", finite)
  print(f"  info  held waist under random flailing (beta {float(act.beta.mean()):.2f}): max {worst:.3f} rad "
        "(motor limit 50 N m; judge it on a trained get-up with eval.py)")
  env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task", help="a HoST-* task, or 'all'")
  parser.add_argument("--num-envs", type=int, default=32)
  args = parser.parse_args()
  for t in list(SPECS) if args.task == "all" else [args.task]:
    run(t, args.num_envs)
  print("ALL PASS" if not failures else f"{len(failures)} FAILED:\n  " + "\n  ".join(failures))
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
