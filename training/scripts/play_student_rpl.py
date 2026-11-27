"""Play a BC or DAgger student checkpoint on one RPL terrain."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import _setup  # noqa: F401  (registers the RPL tasks)
import numpy as np
import torch

from evaluation.student_course import add_student_observations
from g1_comp.rpl import rpl_env_cfg
from g1_comp.student_actor import LegacyTemporalStudentActor, TemporalStudentActor
from g1_comp.student_mapping.observation import StudentMapCfg
from g1_comp.student_distill import FEATURES, TASKS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class StudentMapPolicy:
  """Keep the exact map passed to the student for viewer display."""

  def __init__(self, actor: LegacyTemporalStudentActor | TemporalStudentActor):
    self.actor = actor
    self.current_map: np.ndarray | None = None

  def __call__(self, obs):
    values = obs["student"]
    start = self.actor.STATE_DIM
    self.current_map = values[0, start:start + 33 * 21].detach().cpu().numpy().copy()
    return self.actor(obs)


class StudentMapViserViewer(ViserPlayViewer):
  """Show the student's current 33 x 21 height observation as world points."""

  def setup(self) -> None:
    super().setup()
    self._map_points = self._server.scene.add_point_cloud(
      "/student_height_map", np.zeros((33 * 21, 3), dtype=np.float32),
      colors=(30, 210, 255), point_size=0.025, point_shape="circle", visible=False,
    )
    along = np.linspace(-0.5, 1.1, 33)
    across = np.linspace(-0.5, 0.5, 21)
    x, y = np.meshgrid(along, across, indexing="ij")
    self._map_xy = np.stack((x.ravel(), y.ravel()), axis=-1)

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    values = self.policy.current_map
    if values is None:
      return
    robot = self.env.unwrapped.scene["robot"]
    pelvis_idx = list(robot.body_names).index("pelvis")
    pose = robot.data.body_link_pose_w[0, pelvis_idx].detach().cpu().numpy()
    w, x, y, z = pose[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    xy = self._map_xy @ np.array([[c, s], [-s, c]]) + pose[:2]
    points = np.column_stack((xy, pose[2] - values))
    self._map_points.points = (points + self._scene._scene_offset).astype(np.float32)
    self._map_points.visible = True


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("feature", choices=FEATURES)
  parser.add_argument("--checkpoint", type=Path, required=True,
                      help="BC or DAgger student_*.pt weights checkpoint")
  parser.add_argument("--terrain-level", type=int, choices=range(10), default=9)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--map-mode", choices=("clean", "corrupted"), default="corrupted")
  parser.add_argument("--observation-delay-s", type=float, default=None,
                      help="Override the checkpoint's map delay; use 0 for an immediate map")
  parser.add_argument("--viewer", choices=("viser", "native", "none"), default="viser")
  parser.add_argument("--steps", type=int, default=250, help="Steps with --viewer none")
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Student checkpoint not found: {args.checkpoint}")
  if args.steps < 1:
    parser.error("--steps must be positive")
  if args.observation_delay_s is not None and args.observation_delay_s < 0:
    parser.error("--observation-delay-s must be nonnegative")

  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if saved.get("stage") not in ("bc", "dagger") or "student" not in saved:
    raise ValueError("Expected a BC or DAgger student_*.pt checkpoint")
  obs_dim = saved.get("observation_dim", LegacyTemporalStudentActor.OBS_DIM)
  if obs_dim == LegacyTemporalStudentActor.OBS_DIM:
    actor_cls = LegacyTemporalStudentActor
    map_cfg = StudentMapCfg(**saved["map_config"]) if "map_config" in saved else StudentMapCfg()
    history_length = 2
  elif obs_dim == TemporalStudentActor.OBS_DIM:
    actor_cls = TemporalStudentActor
    map_cfg = StudentMapCfg(**saved["map_config"]) if "map_config" in saved else StudentMapCfg()
    history_length = 5
  else:
    raise ValueError(f"Unsupported student observation size: {obs_dim}")
  if args.observation_delay_s is not None:
    map_cfg = replace(map_cfg, observation_delay_s=args.observation_delay_s)

  os.environ["G1_PLAY_TERRAIN_LEVEL"] = str(args.terrain_level)
  os.environ["G1_PLAY_TERRAIN_SEED"] = str(args.seed)
  torch.manual_seed(args.seed)
  cfg = add_student_observations(rpl_env_cfg(args.feature, play=True), args.map_mode,
                                 history_length=history_length, map_cfg=map_cfg)
  if args.viewer == "viser":
    # The ordinary terrain_scan debug dots are ground truth, even when the
    # student's actual input comes from the corrupted map.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.debug_vis = False
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASKS[args.feature])
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"),
                           clip_actions=agent_cfg.clip_actions)
  try:
    obs = env.get_observations()
    policy = actor_cls(obs, {"student": ["student"]}, "student").to("cpu")
    policy.load_state_dict(saved["student"], strict=True)
    policy.eval()
    print(f"Playing {args.feature} level {args.terrain_level}, seed {args.seed}, "
          f"{args.map_mode} map (delay {map_cfg.observation_delay_s:g}s), "
          f"checkpoint {args.checkpoint}", flush=True)
    if args.viewer == "viser":
      StudentMapViserViewer(env, StudentMapPolicy(policy)).run()
    elif args.viewer == "native":
      NativeMujocoViewer(env, policy).run()
    else:
      with torch.inference_mode():
        for _ in range(args.steps):
          obs, _, _, _ = env.step(policy(obs))
      print(f"Completed {args.steps} steps", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
