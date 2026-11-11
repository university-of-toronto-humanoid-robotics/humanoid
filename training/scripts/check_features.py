"""Check that the sim-to-real training features are wired correctly (no policy needed).

Checks, on G1-Comp-Rough with its training settings:
  1. Mirror: the left/right mapping is its own inverse, the default pose and
     action scales are symmetric, and a mirrored joint pose puts every left body
     exactly where the matching right body was (catches sign mistakes).
  2. Map noise: missing cells, blinks, delays and noise size match the settings.
  3. Assist force, command delay and motor gains are being applied.
  4. Unsafe stepping terms run, and the contact sensor's foot order is [left, right].
Prints PASS/FAIL per check and exits non-zero on any failure. From training/:

  python scripts/check_features.py
"""

import sys
from types import SimpleNamespace

import _setup  # noqa: F401
import torch
from g1_comp import mdp as comp_mdp
from g1_comp import symmetry
from g1_comp.env_cfgs import (
  ASSIST_CUTOFF_LEVEL,
  ASSIST_MAX_SHARE,
  COMMAND_DELAY_STEPS,
  EDGE_HEIGHT,
  MAP_NOISE,
  PD_GAIN_SCALE,
)
from g1_comp.sensors import FOOT_PATCH_RESOLUTION, FOOT_PATCH_SIZE, FOOT_SENSOR_NAMES, grid_shape
from mjlab.actuator import DelayedActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
  if not ok:
    failures.append(name)


def mean(xs: list[float]) -> float:
  return sum(xs) / max(len(xs), 1)


def check_mirror(env: ManagerBasedRlEnv, obs: dict) -> None:
  print("1. Mirror augmentation")
  for group in ("actor", "critic"):
    perm, sign = symmetry.group_mirror(env, group)
    is_perm = sorted(perm.tolist()) == list(range(obs[group].shape[1]))
    twice = (obs[group][:, perm] * sign)[:, perm] * sign
    check(f"{group}: a permutation, and mirroring twice gives the original",
          is_perm and torch.allclose(twice, obs[group]))

  robot = env.scene["robot"]
  jperm, jsign = (torch.tensor(x, device=env.device) for x in symmetry.joint_mirror(list(robot.joint_names)))
  default = robot.data.default_joint_pos[0]
  check("default pose is symmetric", torch.allclose(default[jperm] * jsign, default, atol=1e-6),
        f"max diff {(default[jperm] * jsign - default).abs().max().item():.2e}")
  scale = torch.as_tensor(env.action_manager.get_term("joint_pos")._scale, device=env.device)
  if scale.numel() > 1:
    scale = scale.reshape(-1, env.action_manager.total_action_dim)[0]  # stored per robot
    aperm, _ = symmetry.action_mirror(env)
    check("action scales are symmetric", torch.allclose(scale[aperm], scale))

  # Same root pose for all; robot 1 gets the mirror of robot 0's random pose.
  n = env.num_envs
  q = default.repeat(n, 1)
  q[0] += (torch.rand_like(default) * 2 - 1) * 0.5
  q[1] = q[0][jperm] * jsign
  root = robot.data.default_root_state.clone()
  root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(q, torch.zeros_like(q))
  env.sim.forward()
  pos = robot.data.body_link_pos_w - robot.data.root_link_pos_w.unsqueeze(1)
  names = list(robot.body_names)
  flip_y = torch.tensor([1.0, -1.0, 1.0], device=env.device)
  worst = 0.0
  for i, b in enumerate(names):
    j = names.index(b.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_"))
    worst = max(worst, (pos[1, j] - pos[0, i] * flip_y).abs().max().item())
  check("mirrored pose mirrors every body (joint signs right)", worst < 1e-4, f"worst error {worst * 1000:.3f} mm")

  td = TensorDict(dict(obs.items()), batch_size=[n])
  actions = torch.randn(n, env.action_manager.total_action_dim, device=env.device)
  out_obs, out_act = symmetry.mirror_batch(SimpleNamespace(unwrapped=env), td, actions)
  check("mirror_batch doubles the batch", out_obs.batch_size[0] == 2 * n and out_act.shape[0] == 2 * n)


def check_edge_and_stumble_logic(patch: tuple[int, int]) -> None:
  """The two penalties on hand-made sensor readings (4 cases, one per robot)."""
  rows, cols = patch
  flat = torch.zeros(rows, cols)
  half_off = flat.clone()
  half_off[:, cols // 2 :] = -0.20  # toes over a 20 cm drop
  ramp = torch.arange(cols).float().repeat(rows, 1) * 0.04 * 0.27  # 15 degree slope
  ground = torch.stack([flat, half_off, ramp, half_off]).view(4, -1)
  foot = SimpleNamespace(data=SimpleNamespace(
    distances=torch.ones(4, rows * cols), hit_pos_w=torch.stack([torch.zeros_like(ground)] * 2 + [ground], -1),
    pos_w=torch.full((4, 3), 0.5),
  ))
  in_contact = torch.tensor([[1, 1], [1, 1], [1, 1], [0, 0]])  # last robot: feet in the air
  forces = torch.tensor([[[0.0, 0, 300]] * 2, [[0.0, 0, 300]] * 2, [[0.0, 0, 300]] * 2, [[60.0, 0, 5]] * 2])
  fake = SimpleNamespace(scene={
    "l": foot, "r": foot, "c": SimpleNamespace(data=SimpleNamespace(found=in_contact, force=forces)),
  })
  edge = comp_mdp.foot_on_edge(fake, ("l", "r"), patch, "c", EDGE_HEIGHT)
  stumble = comp_mdp.foot_stumble(fake, "c")
  check("edge penalty: flat 0, over a drop 2, 15 deg ramp 0, in the air 0", edge.tolist() == [0, 2, 0, 0],
        f"got {edge.tolist()}")
  check("stumble penalty: only the sideways push counts", stumble.tolist() == [0, 0, 0, 2], f"got {stumble.tolist()}")


def main() -> None:
  cfg = load_env_cfg("G1-Comp-Rough", play=False)
  cfg.scene.num_envs = 64
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  robot = env.scene["robot"]

  # Collect statistics while stepping (zero actions: robots stand, some fall).
  scan_slice = symmetry.term_slices(env, "actor")["height_scan"]
  height = env.command_manager.get_term("base_height")
  term = env.observation_manager.get_term_cfg("actor", "height_scan").func  # the noisy_height_scan instance
  contact = env.scene["feet_ground_contact"]
  patch = grid_shape(FOOT_PATCH_SIZE, FOOT_PATCH_RESOLUTION)
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  missing, blink, diff, edge, stumble = [], [], [], [], []
  clear_on: list[list[float]] = [[], []]
  clear_off: list[list[float]] = [[], []]
  for _ in range(400):
    obs, *_ = env.step(zero)
    seen = obs["actor"][:, scan_slice]
    fill = height.measured_height().clamp(-1.0, 2.0).unsqueeze(1)
    is_fill = (seen - fill).abs() < 1e-6
    whole = is_fill.all(dim=1)
    blink.append(whole.float().mean().item())
    clean = envs_mdp.height_scan(env, "terrain_scan").clamp(-1.0, 2.0)
    fresh = (term.lag == 0) & ~whole  # older maps were filled with an older pelvis height
    if fresh.any():
      missing.append(is_fill[fresh].float().mean().item())
    if fresh.any():
      diff.append((seen - clean)[fresh][~is_fill[fresh]].abs().mean().item())
    edge.append(comp_mdp.foot_on_edge(env, FOOT_SENSOR_NAMES, patch, "feet_ground_contact", EDGE_HEIGHT).mean().item())
    stumble.append(comp_mdp.foot_stumble(env, "feet_ground_contact").mean().item())
    clearance = comp_mdp.foot_clearance(env, FOOT_SENSOR_NAMES)
    for f in range(2):
      on = contact.data.found[:, f] > 0
      clear_on[f] += clearance[on, f].tolist()
      clear_off[f] += clearance[~on, f].tolist()

  print("2. Map noise")
  expected_blink = MAP_NOISE.blink_rate * sum(MAP_NOISE.blink_steps) / 2
  check("missing cells near the setting", abs(mean(missing) - MAP_NOISE.missing_prob) < 0.01,
        f"{mean(missing) * 100:.2f}% (setting {MAP_NOISE.missing_prob * 100:.0f}%)")
  check("blinks happen, about the expected share", 0 < mean(blink) < 4 * expected_blink,
        f"{mean(blink) * 100:.2f}% of the time (expected ~{expected_blink * 100:.1f}%)")
  lags = term.lag.tolist()
  check("map delays within the setting", MAP_NOISE.lag_steps[0] <= min(lags) and max(lags) <= MAP_NOISE.lag_steps[1],
        f"lags seen {sorted(set(lags))} steps of 20 ms")
  check("noise size plausible (fresh maps)", 0.005 < mean(diff) < MAP_NOISE.point_noise + MAP_NOISE.shift_z + 0.03,
        f"mean |noisy - clean| {mean(diff) * 100:.1f} cm")

  print("3. Assist force, command delay, motor gains")
  pelvis = robot.find_bodies("pelvis")[0]
  gid = int(robot.indexing.body_ids[pelvis[0]])
  applied = env.sim.data.xfrc_applied[:, gid, 2]
  if ASSIST_MAX_SHARE > 0:
    share = comp_mdp.assist_fraction(env, ASSIST_MAX_SHARE, ASSIST_CUTOFF_LEVEL)
    weight = env._g1_comp_robot_weight
    check("assist force is a share of the robot's weight",
          0 <= share <= ASSIST_MAX_SHARE and 0 <= applied.min().item() and applied.max().item() <= ASSIST_MAX_SHARE * weight + 1e-3,
          f"robot weight {weight:.0f} N, share now {share:.2f}, applied {applied.min().item():.0f}-{applied.max().item():.0f} N")
  else:
    check("assist force off (default; G1_COMP_ASSIST=0.5 turns it on)",
          "assist_force" not in env.cfg.events and applied.abs().max().item() == 0.0,
          f"max applied {applied.abs().max().item():.0f} N")
  delayed = [a for a in robot.actuators if isinstance(a, DelayedActuator)]
  check("all actuators delayed", len(delayed) == len(robot.actuators), f"{len(delayed)}/{len(robot.actuators)}")
  lag_sets = torch.stack([a._delay_buffers["position"]._current_lags for a in delayed])
  in_range = COMMAND_DELAY_STEPS[0] <= lag_sets.min().item() and lag_sets.max().item() <= COMMAND_DELAY_STEPS[1]
  check("command delay per robot, same on all joints, within range", in_range and bool((lag_sets == lag_sets[0]).all()),
        f"lags {sorted(set(lag_sets[0].tolist()))} x 5 ms")
  ctrl = torch.cat([torch.as_tensor(a.base_actuator.global_ctrl_ids, device=env.device) for a in delayed]).long()
  ratio = env.sim.model.actuator_gainprm[:, ctrl, 0] / env.sim.get_default_field("actuator_gainprm")[ctrl, 0]
  check("motor gains randomized within range",
        PD_GAIN_SCALE[0] - 1e-4 <= ratio.min().item() and ratio.max().item() <= PD_GAIN_SCALE[1] + 1e-4
        and ratio.std().item() > 0,
        f"Kp scale {ratio.min().item():.3f}-{ratio.max().item():.3f}")

  print("4. Unsafe stepping")
  check("edge and stumble terms run", True, f"feet on edges {mean(edge):.3f}, stumbling {mean(stumble):.3f} per robot-step")
  check_edge_and_stumble_logic(patch)
  for f, side in enumerate(("left", "right")):
    on, off = mean(clear_on[f]), mean(clear_off[f])
    check(f"contact sensor foot {f} is the {side} foot", on < off,
          f"{side} foot clearance {on * 100:.1f} cm in contact vs {off * 100:.1f} cm not")

  check_mirror(env, env.observation_manager.compute())
  env.close()
  print("\nAll checks passed." if not failures else f"\n{len(failures)} FAILED: {', '.join(failures)}")
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
