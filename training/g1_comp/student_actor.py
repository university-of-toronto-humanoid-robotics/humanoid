"""Temporal CNN student; teacher checkpoints keep their original MLP actor."""

from __future__ import annotations

import torch
from torch import nn
from rsl_rl.models import MLPModel


class LegacyTemporalStudentActor(MLPModel):
  """Two-map actor for playback of checkpoints saved before the five-map model."""

  STATE_DIM = 214
  OBS_DIM = STATE_DIM + 2 * 33 * 21

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    groups = obs_groups[obs_set]
    if len(groups) != 1 or obs[groups[0]].shape[-1] != self.OBS_DIM:
      raise ValueError(f"Legacy student expects {self.OBS_DIM} observations")
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


class TemporalStudentActor(MLPModel):
  STATE_DIM = 214
  MAP_COUNT = 5
  MAP_DIM = 33 * 21 * MAP_COUNT
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
      nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Flatten(), nn.Linear(640, 128), nn.ReLU(),
    )
    self.map_positions = nn.Parameter(torch.zeros(1, self.MAP_COUNT, 128))
    nn.init.normal_(self.map_positions, std=0.02)
    layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                       dropout=0.0, batch_first=True, activation="gelu")
    self.map_transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
    self.map_projection = nn.Linear(self.MAP_COUNT * 128, 128)

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    values = obs[self.obs_groups[0]]
    state = self.obs_normalizer(values[:, :self.STATE_DIM])
    maps = values[:, self.STATE_DIM:].reshape(-1, self.MAP_COUNT, 1, 33, 21)
    tokens = self.terrain_encoder(maps.flatten(0, 1)).reshape(-1, self.MAP_COUNT, 128)
    contextual = self.map_transformer(tokens + self.map_positions)
    terrain = self.map_projection(contextual.flatten(1))
    return torch.cat((state, terrain), dim=-1)

  def update_normalization(self, obs):
    if self.obs_normalization:
      self.obs_normalizer.update(obs[self.obs_groups[0]][:, :self.STATE_DIM])
