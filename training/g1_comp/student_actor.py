"""Temporal CNN student; teacher checkpoints keep their original MLP actor."""

from __future__ import annotations

import torch
from torch import nn
from rsl_rl.models import MLPModel


class TemporalStudentActor(MLPModel):
  STATE_DIM = 214
  MAP_DIM = 33 * 21 * 2
  OBS_DIM = STATE_DIM + MAP_DIM

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    groups = obs_groups[obs_set]
    if len(groups) != 1 or obs[groups[0]].shape[-1] != self.OBS_DIM:
      raise ValueError(f"TemporalStudentActor expects one {self.OBS_DIM}-value observation")
    return groups, self.STATE_DIM

  def _get_latent_dim(self) -> int:
    return self.STATE_DIM + 128

  def __init__(self, obs, obs_groups, obs_set, output_dim=29, **kwargs):
    kwargs.setdefault("hidden_dims", (512, 256, 256, 128))
    kwargs.setdefault("activation", "elu")
    kwargs.setdefault("obs_normalization", True)
    kwargs.setdefault("distribution_cfg", None)
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.terrain_encoder = nn.Sequential(
      nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Flatten(), nn.Linear(640, 128), nn.ReLU(),
    )

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    values = obs[self.obs_groups[0]]
    state = self.obs_normalizer(values[:, :self.STATE_DIM])
    maps = values[:, self.STATE_DIM:].reshape(-1, 2, 33, 21)
    return torch.cat((state, self.terrain_encoder(maps)), dim=-1)

  def update_normalization(self, obs):
    if self.obs_normalization:
      self.obs_normalizer.update(obs[self.obs_groups[0]][:, :self.STATE_DIM])
