"""CNN actor for the unchanged 791-value G1 height-map observation."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from rsl_rl.models import MLPModel


class RplCnnActor(MLPModel):
  """Encode the final 693 observations as a 33x21 terrain image.

  The first 98 values retain the existing G1 proprioception and commands.
  The critic remains an independent ordinary MLP on its 806 observations.
  """

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    groups = obs_groups[obs_set]
    if len(groups) != 1 or obs[groups[0]].shape[-1] != 791:
      raise ValueError("RplCnnActor requires one 791-value actor observation group")
    return groups, 98

  def _get_latent_dim(self) -> int:
    return 98 + 128

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.terrain_encoder = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
      nn.Flatten(), nn.Linear(640, 128), nn.ReLU(),
    )

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    values = obs[self.obs_groups[0]]
    state = self.obs_normalizer(values[:, :98])
    terrain = values[:, 98:].reshape(-1, 1, 33, 21)
    return torch.cat((state, self.terrain_encoder(terrain)), dim=-1)

  def update_normalization(self, obs):
    if self.obs_normalization:
      self.obs_normalizer.update(obs[self.obs_groups[0]][:, :98])

  def as_jit(self):
    return _RplCnnExport(self)

  def as_onnx(self, verbose: bool = False):
    del verbose
    return _RplCnnExport(self)


class _RplCnnExport(nn.Module):
  """Deterministic export with the same full observation layout as playback."""

  is_recurrent = False

  def __init__(self, model: RplCnnActor):
    super().__init__()
    self.obs_normalizer = deepcopy(model.obs_normalizer)
    self.terrain_encoder = deepcopy(model.terrain_encoder)
    self.mlp = deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None else nn.Identity()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    state = self.obs_normalizer(x[:, :98])
    terrain = self.terrain_encoder(x[:, 98:].reshape(-1, 1, 33, 21))
    return self.deterministic_output(self.mlp(torch.cat((state, terrain), dim=-1)))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, 791),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass
