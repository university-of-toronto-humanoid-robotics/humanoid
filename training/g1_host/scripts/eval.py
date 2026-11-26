"""Test a stand-up policy, with numbers (and videos). From training/ inside the container:

  # HoST's protocol on one task: 5 repeats x 250 robots, randomization on, no
  # pull force, final action bound 0.25, no observation noise (paper App. A).
  python g1_host/scripts/eval.py --task HoST-Ground --checkpoint logs/rsl_rl/host_ground/<run>/model_<N>.pt

  # The same with robustness sweeps (paper Fig. 5 / Tab. V): payload, constant
  # push from behind, CoM offset forward/back.
  python g1_host/scripts/eval.py --task HoST-Ground --checkpoint <...> --robustness

  # The scenario gallery: every way the robot can be lying, on every surface.
  # Each robot gets the supine or the prone policy from its IMU after the fall
  # (body-frame gravity x < 0: on its back), as the robot will.
  python g1_host/scripts/eval.py --gallery --supine <supine model> --prone <prone model> [--video]

Metrics (HoST's, scripts/eval/eval_ground.py):
  success      pelvis above 0.7 m (slope: 0.6 m above the feet) at some point and
               never back below 0.5 m afterwards
  time         seconds from the end of the fall to first reaching that height
  feet         how much the feet move after standing (cm^2 per step; lower = steadier)
  smoothness   sum of squared second differences of the joint angles (deg, per step)
  energy       sum |torque| |joint speed| dt before standing (J)
Results also go to logs/eval/<name>.json; videos to logs/videos/host/<scenario>.mp4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import _setup  # noqa: E402,F401
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import RecomputeLevel  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from g1_host import mdp, terrains  # noqa: E402
from g1_host import params as P  # noqa: E402
from g1_host.events import Recipe  # noqa: E402
from g1_host.runner import HostRunner  # noqa: E402
from g1_host.tasks import SPECS, Spec, make_env_cfg, runner_cfg  # noqa: E402

DEG, PI = P.DEG, math.pi


def eval_cfg(spec: Spec, num_envs: int, seconds: float, randomize: bool = True):
  cfg = make_env_cfg(spec, play=False)
  cfg.scene.num_envs = num_envs
  act = cfg.actions["joint_pos"]
  act.beta_start, act.pull = P.BETA_EVAL, False
  cfg.observations["actor"].terms["proprio"].params["noise"] = False
  cfg.curriculum = {}
  cfg.episode_length_s = seconds + 2.0  # the run ends before any time-out
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "blew_up"}
  if not randomize:
    cfg.events = {k: v for k, v in cfg.events.items() if k == "reset_posture"}
  return cfg


def load_policy(wrapped, path: str, device: str):
  runner = HostRunner(wrapped, asdict(runner_cfg("HoST-Ground")), device=device)
  runner.load(path, load_cfg={"actor": True}, map_location=device)
  return runner.get_inference_policy(device)


def zero_policy(n, a, device):
  return lambda obs: torch.zeros(n, a, device=device)


def run_episodes(env, wrapped, policies, seconds: float, ref: str, target: float, frames: list | None = None):
  """One episode per robot. policies: (supine, prone or None). Returns per-robot metrics."""
  ent, act = env.scene["robot"], mdp._act(env)
  dev, n = env.device, env.num_envs
  wrapped.reset()
  obs = wrapped.get_observations()
  reached = torch.zeros(n, dtype=torch.bool, device=dev)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  dead = torch.zeros(n, dtype=torch.bool, device=dev)
  use_prone = torch.zeros(n, dtype=torch.bool, device=dev)
  chosen = torch.zeros(n, dtype=torch.bool, device=dev)
  t_stand = torch.full((n,), float("nan"), device=dev)
  feet_move, after = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
  smooth, active = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
  energy = torch.zeros(n, device=dev)
  q_hist: list[torch.Tensor] = []
  last_feet = mdp.feet_pos(env).clone()
  supine, prone = policies
  for _ in range(int(seconds / env.step_dt)):
    with torch.no_grad():
      # Pick the family once, from the IMU, when the robot is done falling.
      newly = act.actuated_after_step & ~chosen
      use_prone[newly] = ent.data.projected_gravity_b[newly, 0] > 0.0
      chosen |= newly
      actions = supine(obs) if prone is None else torch.where(use_prone.unsqueeze(1), prone(obs), supine(obs))
      obs, _, dones, _ = wrapped.step(actions)
    dead |= dones.bool()
    h = mdp.base_height(env, ref)
    first = (h > target) & ~reached & ~dead
    t_stand[first] = ((act.steps - act.unactuated).float() * env.step_dt)[first]
    reached |= first
    fell |= reached & (h < 0.5)
    feet = mdp.feet_pos(env)
    feet_move += reached.float() * ((feet - last_feet) * 100).square().sum(dim=(1, 2))
    after += reached.float()
    last_feet = feet.clone()
    on = act.actuated.float()
    q_hist = (q_hist + [mdp._q(env).clone()])[-3:]
    if len(q_hist) == 3:
      smooth += on * (q_hist[2] - 2 * q_hist[1] + q_hist[0]).square().sum(dim=1)
      active += on
    energy += on * (~reached).float() * (mdp._torque(env).abs() * mdp._qd(env).abs()).sum(dim=1) * env.step_dt
    if frames is not None:
      frames.append(env.render())
  success = reached & ~fell & ~dead
  return {"success": success.float(), "time": t_stand, "feet": feet_move / after.clamp(min=1),
          "smoothness": smooth / active.clamp(min=1) * 180 / PI, "energy": energy, "prone_policy": use_prone.float()}


def summarize(per_repeat: list[dict]) -> dict:
  out = {}
  for key in ("success", "time", "feet", "smoothness", "energy", "prone_policy"):
    vals = []
    for r in per_repeat:
      x = r[key]
      if key in ("feet", "time"):
        x = x[r["success"] > 0]
      x = x[~torch.isnan(x)]
      vals.append(float(x.mean()) if len(x) else float("nan"))
    t = torch.tensor(vals)
    scale = 100.0 if key in ("success", "prone_policy") else 1.0
    std = float(t[~torch.isnan(t)].std()) if int((~torch.isnan(t)).sum()) > 1 else 0.0
    out[key] = [float(t.nanmean()) * scale, std * scale]
  return out


def fmt(s: dict) -> str:
  return (f"success {s['success'][0]:5.1f}% (+-{s['success'][1]:.1f})  time {s['time'][0]:.2f}s  "
          f"feet {s['feet'][0]:.2f}  smooth {s['smoothness'][0]:.2f}  energy {s['energy'][0]:.0f}J  "
          f"prone-policy {s['prone_policy'][0]:.0f}%")


def build(cfg, device: str, video: bool):
  if video:
    cfg.viewer = replace(cfg.viewer, height=480, width=640, max_extra_envs=0)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if video else None)
  return env, RslRlVecEnvWrapper(env)


def save_video(frames: list, name: str, fps: float) -> None:
  import imageio.v2 as imageio

  out = Path("logs/videos/host")
  out.mkdir(parents=True, exist_ok=True)
  with imageio.get_writer(out / f"{name}.mp4", fps=round(fps)) as w:
    for f in frames:
      w.append_data(f)
  print(f"    video: {out / (name + '.mp4')}", flush=True)


# --- The gallery ---------------------------------------------------------------------


def _one(kind: str, sub, **recipe) -> Spec:
  world = None if sub is None else terrains.world({kind: sub}, num_rows=1, cols_per_type=1)
  return Spec(kinds=(kind,), recipes=(Recipe(**recipe),), world=world, ref="ground")


def _slope(deg: float):
  return terrains.SlopeTerrainCfg(slopes=(math.tan(deg * DEG),))


def _stairs():
  return terrains.StairsTerrainCfg(riser_range=(0.203, 0.203))  # the course's


_OURS = {"unactuated": P.UNACTUATED_STEPS_OURS}
_WALL = {"postures": {"sit": 1.0}, "drop": 0.45, "pose": P.RESET_POSE_WALL, "unactuated": P.UNACTUATED_STEPS_WALL,
         "pushback": P.WALL_PUSHBACK_N}

GALLERY: dict[str, Spec] = {
  "flat_supine": _one("ground", None, postures={"supine": 1.0}, yaw_noise=PI),
  "flat_prone": _one("ground", None, postures={"prone": 1.0}, yaw_noise=PI),
  "flat_side_left": _one("ground", None, postures={"side_left": 1.0}, yaw_noise=PI),
  "flat_side_right": _one("ground", None, postures={"side_right": 1.0}, yaw_noise=PI),
  "flat_random_fall": _one("ground", None, postures={"random": 1.0}, **_OURS),
  "platform_30cm": _one("platform", terrains.PlatformTerrainCfg(height_range=(0.30, 0.30)), postures={"supine": 1.0},
                        pose=P.RESET_POSE_PLATFORM),
  "wall_30deg": _one("wall", terrains.WallTerrainCfg(slopes=(math.tan(30 * DEG),)), **_WALL),
  "wall_60deg": _one("wall", terrains.WallTerrainCfg(slopes=(math.tan(60 * DEG),)), **_WALL),
  "slope_10deg_head_up": _one("slope", _slope(10), postures={"supine": 1.0}, yaws=(0.0,)),
  "slope_10deg_head_down": _one("slope", _slope(10), postures={"supine": 1.0}, yaws=(PI,)),
  "ramp_15deg_head_up": _one("slope", _slope(15), postures={"supine": 1.0}, yaws=(0.0,)),
  "ramp_15deg_head_down": _one("slope", _slope(15), postures={"supine": 1.0}, yaws=(PI,)),
  "ramp_15deg_sideways": _one("slope", _slope(15), postures={"supine": 1.0}, yaws=(PI / 2, -PI / 2)),
  "ramp_15deg_prone_head_down": _one("slope", _slope(15), postures={"prone": 1.0}, yaws=(PI,)),
  "stairs_head_up": _one("stairs", _stairs(), postures={"supine": 1.0}, yaws=(0.0,), drop=0.6, **_OURS),
  "stairs_head_down": _one("stairs", _stairs(), postures={"supine": 1.0}, yaws=(PI,), drop=0.6, **_OURS),
  "stairs_sideways": _one("stairs", _stairs(), postures={"supine": 1.0}, yaws=(PI / 2, -PI / 2), drop=0.6, **_OURS),
  "stairs_prone_head_down": _one("stairs", _stairs(), postures={"prone": 1.0}, yaws=(PI,), drop=0.6, **_OURS),
  "stairs_random_fall": _one("stairs", _stairs(), postures={"random": 1.0}, **_OURS),
}


def _robustness(env, wrapped, pol, seconds, ref, target, results) -> None:
  dev = env.device
  ent = env.scene["robot"]
  tb = int(ent.indexing.body_ids[ent.find_bodies("torso_link")[0][0]])
  mass0 = env.sim.model.body_mass[:, tb].clone()
  ipos0 = env.sim.model.body_ipos[:, tb].clone()

  def measure(label: str) -> None:
    s = summarize([run_episodes(env, wrapped, (pol, None), seconds, ref, target) for _ in range(2)])
    print(f"  {label:18s} {fmt(s)}", flush=True)
    results[label] = s

  for kg in (4, 8, 12):
    env.sim.model.body_mass[:, tb] = mass0 + kg
    env.sim.recompute_constants(RecomputeLevel.set_const)
    measure(f"payload +{kg} kg")
  env.sim.model.body_mass[:, tb] = mass0
  for dx in (-0.1, -0.05, 0.05, 0.1):
    env.sim.model.body_ipos[:, tb] = ipos0 + torch.tensor([dx, 0.0, 0.0], device=dev)
    env.sim.recompute_constants(RecomputeLevel.set_const)
    measure(f"CoM x {dx:+.2f} m")
  env.sim.model.body_ipos[:, tb] = ipos0
  env.sim.recompute_constants(RecomputeLevel.set_const)
  act = mdp._act(env)
  for fx in (25, 50, 75, 100):
    act.extra_force[:] = torch.tensor([float(fx), 0.0, 0.0], device=dev)
    measure(f"push +x {fx} N")
  act.extra_force[:] = 0.0


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--task", help="HoST task for the paper protocol (e.g. HoST-Ground)")
  ap.add_argument("--checkpoint", help="model_<N>.pt for --task; leave out for zero actions")
  ap.add_argument("--run", help="instead of --checkpoint: the newest model of the task's run with this name")
  ap.add_argument("--gallery", action="store_true")
  ap.add_argument("--supine", help="gallery: the supine-family model")
  ap.add_argument("--prone", help="gallery: the prone-family model (else the supine one everywhere)")
  ap.add_argument("--only", nargs="*", help="gallery: only these scenarios (names: see GALLERY)")
  ap.add_argument("--robots", type=int, default=250)
  ap.add_argument("--repeats", type=int, default=5)
  ap.add_argument("--seconds", type=float, default=10.0, help="per episode, including the fall (HoST: 5)")
  ap.add_argument("--robustness", action="store_true")
  ap.add_argument("--video", action="store_true", help="also record robot 0 of the first repeat")
  ap.add_argument("--no-randomization", action="store_true")
  ap.add_argument("--child", help=argparse.SUPPRESS)  # internal: one gallery scenario, result to this file
  args = ap.parse_args()
  if not (args.task or args.gallery):
    ap.error("give --task and/or --gallery")
  if args.run:
    from _runs import latest_checkpoint

    from g1_host.tasks import experiment_name

    found = latest_checkpoint(experiment_name(args.task), args.run)
    if found is None:
      raise SystemExit(f"no checkpoint of run '{args.run}' for {args.task} under logs/rsl_rl/")
    args.checkpoint = str(found[0])
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"
  results: dict = {}

  if args.task:
    spec = SPECS[args.task]
    target = 0.6 if spec.ref == "feet" else 0.7
    env, wrapped = build(eval_cfg(spec, args.robots, args.seconds, not args.no_randomization), dev, args.video)
    a = mdp._act(env).action_dim
    pol = load_policy(wrapped, args.checkpoint, dev) if args.checkpoint else zero_policy(env.num_envs, a, dev)
    print(f"{args.task}: {args.repeats} x {env.num_envs} robots, {args.seconds:.0f} s each", flush=True)
    reps = []
    for r in range(args.repeats):
      frames = [] if (args.video and r == 0) else None
      reps.append(run_episodes(env, wrapped, (pol, None), args.seconds, spec.ref, target, frames))
      if frames:
        save_video(frames, args.task, 1 / env.step_dt)
    s = summarize(reps)
    print(f"  {'nominal':18s} {fmt(s)}", flush=True)
    results["nominal"] = s
    if args.robustness:
      _robustness(env, wrapped, pol, args.seconds, spec.ref, target, results)
    env.close()

  if args.gallery and not args.child:
    # One process per scenario: building many simulators in one process runs out of
    # Warp's GPU state (CUDA graph capture fails after ~10).
    names = args.only or list(GALLERY)
    print(f"gallery: {len(names)} scenarios x {args.robots} robots x {args.repeats} repeats", flush=True)
    common = ["--gallery", "--robots", str(args.robots), "--repeats", str(args.repeats), "--seconds", str(args.seconds)]
    for flag, value in (("--supine", args.supine), ("--prone", args.prone)):
      if value:
        common += [flag, value]
    common += ["--video"] * args.video + ["--no-randomization"] * args.no_randomization
    for name in names:
      frag = Path("logs/eval") / f"_scenario_{name}.json"
      subprocess.run([sys.executable, os.path.abspath(__file__), *common, "--only", name, "--child", str(frag)],
                     check=False)
      if frag.is_file():
        results[name] = json.loads(frag.read_text())
        frag.unlink()
      else:
        results[name] = {"error": "scenario failed, see the output above"}
  elif args.gallery:
    for name in args.only:
      env, wrapped = build(eval_cfg(GALLERY[name], args.robots, args.seconds, not args.no_randomization), dev,
                           args.video)
      a = mdp._act(env).action_dim
      sup = load_policy(wrapped, args.supine, dev) if args.supine else zero_policy(env.num_envs, a, dev)
      pro = load_policy(wrapped, args.prone, dev) if args.prone else None
      reps = []
      for r in range(args.repeats):
        frames = [] if (args.video and r == 0) else None
        reps.append(run_episodes(env, wrapped, (sup, pro), args.seconds, "ground", 0.7, frames))
        if frames:
          save_video(frames, name, 1 / env.step_dt)
      s = summarize(reps)
      print(f"  {name:28s} {fmt(s)}", flush=True)
      Path(args.child).parent.mkdir(parents=True, exist_ok=True)
      Path(args.child).write_text(json.dumps(s))
      env.close()
    return

  out = Path("logs/eval")
  out.mkdir(parents=True, exist_ok=True)
  tag = Path(args.checkpoint or args.supine or "zero").stem
  name = f"{args.task or 'gallery'}{'_gallery' if args.task and args.gallery else ''}_{tag}"
  (out / f"{name}.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
  print(f"wrote {out / (name + '.json')}")


if __name__ == "__main__":
  main()
