"""Bounded, CPU-backed action imitation replay with equal terrain sampling."""

from __future__ import annotations

import torch

from .student_actor import TemporalStudentActor
from .student_distill import FEATURES


class Ring:
  def __init__(self, capacity: int):
    if capacity < 1:
      raise ValueError("Replay capacity must be positive")
    self.capacity = capacity
    self.observations = None
    self.actions = None
    self.count = 0
    self.pointer = 0

  def __len__(self):
    return self.count

  def add(self, observations: torch.Tensor, actions: torch.Tensor):
    if observations.ndim != 2 or observations.shape[1] != TemporalStudentActor.OBS_DIM:
      raise ValueError("Expected 1600-value student observations")
    if actions.shape != (len(observations), 29):
      raise ValueError("Expected matching 29-value teacher actions")
    if not len(observations):
      return
    if self.observations is None:
      self.observations = torch.empty((self.capacity, TemporalStudentActor.OBS_DIM), dtype=torch.float16)
      self.actions = torch.empty((self.capacity, 29), dtype=torch.float16)
    observations = observations.detach()[-self.capacity:].to(device="cpu", dtype=torch.float16)
    actions = actions.detach()[-self.capacity:].to(device="cpu", dtype=torch.float16)
    positions = (torch.arange(len(observations)) + self.pointer) % self.capacity
    self.observations[positions] = observations
    self.actions[positions] = actions
    self.pointer = (self.pointer + len(observations)) % self.capacity
    self.count = min(self.capacity, self.count + len(observations))

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.count:
      raise ValueError("Cannot sample an empty replay bucket")
    index = torch.randint(self.count, (count,))
    return self.observations[index], self.actions[index]

  def state_dict(self):
    return {"capacity": self.capacity, "count": self.count, "pointer": self.pointer,
            "observations": self.observations[:self.count].clone() if self.count else None,
            "actions": self.actions[:self.count].clone() if self.count else None}

  def load_state_dict(self, state):
    if state["capacity"] != self.capacity:
      raise ValueError("Replay capacity changed across resume")
    self.count, self.pointer = state["count"], state["pointer"]
    if self.count:
      self.observations = torch.empty((self.capacity, TemporalStudentActor.OBS_DIM), dtype=torch.float16)
      self.actions = torch.empty((self.capacity, 29), dtype=torch.float16)
      self.observations[:self.count] = state["observations"]
      self.actions[:self.count] = state["actions"]


def _split_count(total: int, buckets: int) -> list[int]:
  counts = [total // buckets] * buckets
  for index in torch.randperm(buckets)[:total % buckets].tolist():
    counts[index] += 1
  return counts


class StudentReplay:
  def __init__(self, per_terrain_capacity: int):
    self.per_terrain_capacity = per_terrain_capacity
    self.bc = {feature: [Ring(max(1, per_terrain_capacity // 10)) for _ in range(10)]
               for feature in FEATURES}
    self.dagger = {feature: Ring(per_terrain_capacity) for feature in FEATURES}

  def add_bc(self, feature: str, obs: torch.Tensor, actions: torch.Tensor, levels: torch.Tensor):
    for level in range(10):
      mask = levels == level
      if mask.any():
        self.bc[feature][level].add(obs[mask], actions[mask])

  def add_dagger(self, feature: str, obs: torch.Tensor, actions: torch.Tensor):
    self.dagger[feature].add(obs, actions)

  def sample(self, batch_size: int, bc_fraction: float, device: str):
    if batch_size < 1 or not 0 <= bc_fraction <= 1:
      raise ValueError("Invalid batch size or BC fraction")
    bc_count = round(batch_size * bc_fraction)
    if bc_fraction == 1:
      bc_count = batch_size
    terrain_bc = _split_count(bc_count, 4)
    terrain_dagger = _split_count(batch_size - bc_count, 4)
    parts = []
    for i, feature in enumerate(FEATURES):
      available = [ring for ring in self.bc[feature] if len(ring)]
      if terrain_bc[i]:
        if len(available) != 10:
          raise RuntimeError(f"BC replay lacks one or more levels for {feature}")
        for ring, n in zip(available, _split_count(terrain_bc[i], 10), strict=True):
          if n:
            parts.append(ring.sample(n))
      if terrain_dagger[i]:
        parts.append(self.dagger[feature].sample(terrain_dagger[i]))
    obs = torch.cat([part[0] for part in parts]).to(device=device, dtype=torch.float32)
    actions = torch.cat([part[1] for part in parts]).to(device=device, dtype=torch.float32)
    order = torch.randperm(len(obs), device=device)
    return obs[order], actions[order]

  def bc_coverage(self):
    return {feature: [len(ring) for ring in self.bc[feature]] for feature in FEATURES}

  def state_dict(self):
    return {"capacity": self.per_terrain_capacity,
            "bc": {f: [ring.state_dict() for ring in self.bc[f]] for f in FEATURES},
            "dagger": {f: self.dagger[f].state_dict() for f in FEATURES}}

  def load_state_dict(self, state, *, include_dagger: bool = True):
    if state["capacity"] != self.per_terrain_capacity:
      raise ValueError("Replay capacity must match the checkpoint")
    for feature in FEATURES:
      for ring, saved in zip(self.bc[feature], state["bc"][feature], strict=True):
        ring.load_state_dict(saved)
      if include_dagger:
        self.dagger[feature].load_state_dict(state["dagger"][feature])

  def load_bc_demos(self, state):
    """Import BC examples into this run's capacity without importing its optimizer."""
    for feature in FEATURES:
      for ring, saved in zip(self.bc[feature], state["bc"][feature], strict=True):
        if saved["count"]:
          ring.add(saved["observations"], saved["actions"])
