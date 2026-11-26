"""Student observation variants on the existing competition course."""

from __future__ import annotations

from copy import deepcopy

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import GaussianNoiseCfg

from g1_comp.env_cfgs import g1_course_heightmap_env_cfg
from g1_comp.student_mapping.observation import StudentHeightMap, StudentMapCfg, student_probe_cfg


def clean_student_height_scan(env, sensor_name: str) -> torch.Tensor:
  """Perfect current terrain scan in both temporal channels of the student CNN.

  This is a diagnostic upper bound: both channels see today's exact terrain,
  rather than a delayed or incomplete map from an earlier time.
  """
  scan = envs_mdp.height_scan(env, sensor_name).clamp(-1.0, 2.0)
  if scan.shape[-1] != 33 * 21:
    raise ValueError(f"Expected a 33 x 21 terrain scan, got {scan.shape[-1]} values")
  scan = scan.reshape(-1, 21, 33).transpose(1, 2).reshape(-1, 33 * 21)
  return torch.cat((scan, scan), dim=-1)


def add_student_observations(cfg, map_mode: str = "corrupted"):
  """Add the trained student's observations to a compatible height-map environment."""
  if map_mode not in ("clean", "corrupted"):
    raise ValueError("map_mode must be clean or corrupted")
  student = deepcopy(cfg.observations["actor"])
  student.history_length = None
  student.terms["joint_pos"].history_length = 5
  if map_mode == "corrupted":
    student.enable_corruption = True
    student.terms["joint_pos"].noise = GaussianNoiseCfg(mean=0.0, std=0.01)
    student.terms["joint_vel"].noise = GaussianNoiseCfg(mean=0.0, std=0.5)
    map_cfg = StudentMapCfg()
    student.terms["height_scan"] = ObservationTermCfg(
      func=StudentHeightMap,
      params={"sensor_name": "student_map_probe", "map_cfg": map_cfg, "return_history": True},
      clip=(-1.0, 2.0),
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (student_probe_cfg(map_cfg),)
  else:
    student.enable_corruption = False
    student.terms["height_scan"] = ObservationTermCfg(
      func=clean_student_height_scan, params={"sensor_name": "terrain_scan"},
      clip=(-1.0, 2.0),
    )
  cfg.observations["student"] = student
  return cfg


def student_course_env_cfg(map_mode: str = "corrupted"):
  """Preserve the competition course and route, adding student observations."""
  return add_student_observations(g1_course_heightmap_env_cfg(play=True), map_mode)
