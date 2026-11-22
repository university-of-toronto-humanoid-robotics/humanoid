"""Deterministic checks for the RPL foot-force and sole-support decisions."""

import unittest

import _setup  # noqa: F401
import torch

from g1_comp.mdp import foothold_from_samples, stumble_from_forces


class RplSafetyTests(unittest.TestCase):
  def test_stumble_force_cases(self):
    cases = (
      ("air", [[0, 0, 0], [0, 0, 0]], 0.0),
      ("ordinary landing", [[2, 0, 200], [0, 1, 150]], 0.0),
      ("left toe into riser", [[20, 0, 2], [0, 0, 100]], 1.0),
      ("right toe into riser", [[0, 0, 100], [0, 20, 2]], 1.0),
      ("both feet", [[20, 0, 2], [0, 20, 2]], 1.0),
      ("solver noise", [[1, 0, 0], [0, 0, 0]], 0.0),
    )
    for name, forces, expected in cases:
      with self.subTest(name=name):
        actual = stumble_from_forces(torch.tensor([forces], dtype=torch.float), 4.0, 5.0)
        self.assertEqual(actual.item(), expected)

  def test_ratio_is_explicit(self):
    forces = torch.tensor([[[30.0, 0.0, 20.0], [0.0, 0.0, 0.0]]])
    self.assertEqual(stumble_from_forces(forces, 0.5, 5.0).item(), 1.0)
    self.assertEqual(stumble_from_forces(forces, 4.0, 5.0).item(), 0.0)

  def test_foothold_support_and_air(self):
    heights = torch.zeros((1, 2, 50))
    ground = torch.zeros_like(heights)
    both_contact = torch.tensor([[[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]])
    value, count = foothold_from_samples(both_contact, heights, ground, 0.05, 1.0)
    self.assertEqual((value.item(), count.item()), (0.0, 0.0))

    ground[0, 0, :25] = -0.10
    ground[0, 1, :] = -0.10
    left_contact = both_contact.clone()
    left_contact[0, 1, 2] = 0.0
    value, count = foothold_from_samples(left_contact, heights, ground, 0.05, 1.0)
    self.assertEqual((value.item(), count.item()), (0.5, 25.0))
    value, count = foothold_from_samples(torch.zeros_like(both_contact), heights, ground, 0.05, 1.0)
    self.assertEqual((value.item(), count.item()), (0.0, 0.0))
    value, count = foothold_from_samples(both_contact, heights, ground, 0.05, 1.0)
    self.assertEqual((value.item(), count.item()), (1.5, 75.0))


if __name__ == "__main__":
  unittest.main()
