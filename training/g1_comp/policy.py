"""PPO for the new tasks: mirror augmentation, plus fixed input scales for the
inputs that differ between tasks.

RSL-RL rescales every input by its running mean and spread. That breaks when a
model moves between tasks:

  - On flat ground the height map never changes, so its spread is ~4 cm. On stairs
    the same input then reads 5-20x larger than anything the model has seen.
  - The crouch command and the arm flag are constant in tasks without crouching
    or arm motion: spread 0, so the first real crouch command reads as -36.

So those inputs get a fixed mean and spread (PINNED), the same in every task. A
model trained on Flat then sees stairs, a crouch command or an arm takeover as
ordinary numbers. The other inputs keep the running statistics.
"""

from __future__ import annotations

import os

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization

from .symmetry import term_slices

# Observation term -> (mean, spread). The height map is terrain height relative to
# the feet in meters (scan.py), passed through unscaled: 0 on flat ground, 0.2 for
# a 20 cm step, at most 0.9 for a wall. Kept small on purpose: a wall fills hundreds
# of cells with the same value and the optimizer moves all their weights together,
# so a larger scale made a new network's outputs jump on its first update.
PINNED = {
  "height_scan": (0.0, 1.0),
  "height_command": (0.6, 0.2),
  "arm_override": (0.0, 1.0),
}
# A new network starts with zero weight on these inputs. On Flat the map carries
# nothing (0 plus noise), so the weights stay near zero, and the first stairs don't
# disturb a model that learned to walk on Flat: it starts using the map gradually.
START_BLIND_TO = ("height_scan",)
# Mirror augmentation (symmetry.py) on by default; G1_MIRROR=0 turns it off to compare.
MIRROR = os.environ.get("G1_MIRROR", "1") != "0"


class PinnedNormalization(EmpiricalNormalization):
  """RSL-RL's running normalizer, with some inputs held at a fixed mean and spread.
  Saved and loaded exactly like the original (same buffers)."""

  def __init__(self, shape: int, pins: list[tuple[slice, float, float]], eps: float = 1e-2) -> None:
    super().__init__(shape, eps)
    mask = torch.zeros(1, shape, dtype=torch.bool)
    mean = torch.zeros(1, shape)
    std = torch.ones(1, shape)
    for where, m, s in pins:
      mask[0, where] = True
      mean[0, where] = m
      std[0, where] = s - eps  # forward() divides by (std + eps)
    self.register_buffer("pin_mask", mask, persistent=False)
    self.register_buffer("pin_mean", mean, persistent=False)
    self.register_buffer("pin_std", std, persistent=False)
    self.pin()

  @torch.jit.unused
  def pin(self) -> None:
    self._mean = torch.where(self.pin_mask, self.pin_mean, self._mean)
    self._std = torch.where(self.pin_mask, self.pin_std, self._std)
    self._var = torch.where(self.pin_mask, self.pin_std**2, self._var)

  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    super().update(x)
    self.pin()


def pin_normalizers(alg: PPO, env, device: str) -> None:
  base = env.unwrapped
  for model, group in ((alg.actor, "actor"), (alg.critic, "critic")):
    if not isinstance(model.obs_normalizer, EmpiricalNormalization):
      continue
    slices = term_slices(base, group)
    assert max(s.stop for s in slices.values()) == model.obs_dim, f"{group}: observation size mismatch"
    pins = [(where, *PINNED[name]) for name, where in slices.items() if name in PINNED]
    model.obs_normalizer = PinnedNormalization(model.obs_dim, pins).to(device)
    # Only matters for a new network: loading a saved model replaces these weights.
    with torch.no_grad():
      for name in START_BLIND_TO:
        if name in slices:
          model.mlp[0].weight[:, slices[name]] = 0.0


class CompPPO(PPO):
  """RSL-RL's PPO with mirror augmentation and pinned input scales. Used by
  setting the algorithm's class_name, so no new config fields are needed."""

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):  # type: ignore[override]
    if MIRROR:
      cfg["algorithm"]["symmetry_cfg"] = {
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "data_augmentation_func": "g1_comp.symmetry:mirror_batch",
      }
    alg = PPO.construct_algorithm(obs, env, cfg, device)
    # As in symmetry.MirrorPPO: keep the run config writable to params/agent.yaml.
    cfg["algorithm"].pop("symmetry_cfg", None)
    cfg["algorithm"]["class_name"] = "g1_comp.policy:CompPPO"
    pin_normalizers(alg, env, device)
    return alg
