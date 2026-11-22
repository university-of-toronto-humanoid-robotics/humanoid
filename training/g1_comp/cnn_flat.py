"""Flat height-map task with the same CNN actor as RPL-CNN-Ascend-Stairs."""

from .env_cfgs import g1_flat_heightmap_ppo_runner_cfg

TASK = "G1-CNN-Flat-HeightMap"


def cnn_flat_runner_cfg():
  cfg = g1_flat_heightmap_ppo_runner_cfg()
  cfg.experiment_name = "g1_cnn_flat_heightmap"
  cfg.actor.class_name = "g1_comp.rpl_cnn_actor:RplCnnActor"
  cfg.actor.hidden_dims = (512, 256, 256, 128)
  return cfg
