"""Observation contract and terrain setup for student imitation."""

from __future__ import annotations

from copy import deepcopy

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import GaussianNoiseCfg

from .env_cfgs import g1_flat_heightmap_env_cfg
from .rpl import rpl_env_cfg, rpl_maze_env_cfg
from .student_mapping.observation import StudentHeightMap, StudentMapCfg, student_probe_cfg

FEATURES = ("ascend", "descend", "ramps", "obstacles")
FEATURES_V2 = (*FEATURES, "maze", "flat")
TASKS = {
  "ascend": "RPL-Ascend-Stairs",
  "descend": "RPL-Descend-Stairs",
  "ramps": "RPL-Ramps",
  "obstacles": "RPL-Obstacles",
  "maze": "RPL-Maze",
  "flat": "G1-Flat-HeightMap",
}


def randomize_bc_terrain(env, env_ids: torch.Tensor) -> None:
  """Choose a level independently at each reset, before root placement."""
  terrain = env.scene.terrain
  assert terrain is not None
  terrain.randomize_env_origins(env_ids)
  if len(env_ids) >= 10:
    # A large reset should cover all rows, including the first one at startup.
    terrain.terrain_levels[env_ids[:10]] = torch.randperm(10, device=env.device)
    terrain.env_origins[env_ids[:10]] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids[:10]], terrain.terrain_types[env_ids[:10]]
    ]


def restore_terrain_levels(env, env_ids: torch.Tensor | None, levels: list[int]) -> None:
  """Restore difficulty assignments before the wrapper's first reset."""
  del env_ids
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  values = torch.as_tensor(levels, dtype=torch.long, device=env.device)
  if values.shape != terrain.terrain_levels.shape:
    raise ValueError("Full resume requires the original environment count")
  terrain.terrain_levels[:] = values
  terrain.env_origins[:] = terrain.terrain_origins[values, terrain.terrain_types]


def student_env_cfg(feature: str, *, stage: str, num_envs: int,
                    map_cfg: StudentMapCfg | None = None,
                    resume_levels: list[int] | None = None):
  """Keep the original actor group for the frozen expert; add student separately."""
  if feature not in FEATURES_V2 or stage not in ("bc", "dagger"):
    raise ValueError("Unknown feature or distillation stage")
  cfg = (rpl_maze_env_cfg() if feature == "maze" else
         g1_flat_heightmap_env_cfg() if feature == "flat" else
         rpl_env_cfg(feature, play=False))
  cfg.scene.num_envs = num_envs
  cfg.observations["actor"].enable_corruption = False
  if "corrupt" in cfg.observations["actor"].terms["height_scan"].params:
    cfg.observations["actor"].terms["height_scan"].params["corrupt"] = False

  student = deepcopy(cfg.observations["actor"])
  student.enable_corruption = True
  student.history_length = None  # Joint history is managed here; map history is in StudentHeightMap.
  student.terms["joint_pos"].history_length = 5
  student.terms["joint_pos"].noise = GaussianNoiseCfg(mean=0.0, std=0.01)
  student.terms["joint_vel"].noise = GaussianNoiseCfg(mean=0.0, std=0.5)
  map_cfg = map_cfg or StudentMapCfg()
  student.terms["height_scan"] = ObservationTermCfg(
    func=StudentHeightMap,
    params={"sensor_name": "student_map_probe", "map_cfg": map_cfg, "return_history": True},
    clip=(-1.0, 2.0),
  )
  cfg.observations["student"] = student
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (student_probe_cfg(map_cfg),)

  if stage == "bc" and feature != "flat":
    cfg.curriculum.pop("terrain_levels", None)
    cfg.scene.terrain.max_init_terrain_level = 9
    events = dict(cfg.events)
    events.pop("randomize_terrain", None)
    cfg.events = {"randomize_terrain": EventTermCfg(func=randomize_bc_terrain, mode="reset"), **events}
  elif resume_levels is not None and feature != "flat":
    cfg.events["restore_student_levels"] = EventTermCfg(
      func=restore_terrain_levels, mode="startup", params={"levels": resume_levels}
    )
  return cfg
