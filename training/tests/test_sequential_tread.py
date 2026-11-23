"""Deterministic checks of ordered stair-descent contact credit."""

import unittest
from types import SimpleNamespace as NS

import _setup  # noqa: F401
import torch

from g1_comp import worlds
from g1_comp.sequential_tread import SequentialTreadReward


class Scene(dict):
  pass


class SequentialTreadTests(unittest.TestCase):
  def setUp(self):
    sub = worlds.RplLaneTerrainCfg(feature="descend")
    sub.stair_specs.append((3, 0.1, 0.3, 0.8))
    generator = NS(num_rows=1, num_cols=1, sub_terrains={"descend": sub})
    terrain = NS(cfg=NS(terrain_generator=generator),
                 terrain_levels=torch.tensor([0]), terrain_types=torch.tensor([0]),
                 env_origins=torch.tensor([[0.45, 1.6, 0.3]]))
    scene = Scene(terrain=terrain)
    scene.terrain = terrain
    scene["robot"] = NS(indexing=NS(root_body_id=1),
                        data=NS(root_link_pos_w=torch.tensor([[0.45, 1.6, 0.3]])))
    for side in ("left", "right"):
      scene[f"{side}_sole_support"] = NS(
        sole_points_w=torch.zeros((1, 50, 3)),
        data=NS(distances=torch.zeros((1, 50)), hit_pos_w=torch.zeros((1, 50, 3))),
      )
    scene["feet_environment_contact"] = NS(data=NS(force=torch.zeros((1, 2, 3))))
    self.env = NS(scene=scene, sim=NS(mj_model=NS(body_subtreemass=[0.0, 40.0])),
                  device="cpu", num_envs=1, step_dt=0.02,
                  _lane_start=torch.tensor([[0.45, 1.6]]), extras={"log": {}})
    self.term = SequentialTreadReward(None, self.env)

  def place(self, foot=None, tread=0):
    for side in ("left", "right"):
      sensor = self.env.scene[f"{side}_sole_support"]
      sensor.sole_points_w[:] = torch.tensor([0.45, 1.6, 0.3])
      sensor.data.hit_pos_w[..., 2] = 0.3
    force = self.env.scene["feet_environment_contact"].data.force
    force[:] = 0
    if foot is not None:
      x = (2.15, 2.45, 3.20)[tread]
      z = (0.2, 0.1, 0.0)[tread]
      side = ("left", "right")[foot]
      sensor = self.env.scene[f"{side}_sole_support"]
      sensor.sole_points_w[:] = torch.tensor([x, 1.6, z])
      sensor.data.hit_pos_w[..., 2] = z
      force[0, foot, 2] = -100.0

  def step(self):
    return self.term(self.env, 0.50, 0.10, 0.5, 0.5, 0.5, 0.30)

  def test_alternating_contacts_and_clean_finish(self):
    for tread, foot in ((0, 0), (1, 1), (2, 0)):
      self.place(foot, tread)
      for _ in range(5):
        reward = self.step()
      self.assertGreater(reward.item(), 20.0)  # One-time dt-scaled completion bonus.
      self.assertEqual(self.term.target.item(), tread + 1)
    self.place()
    self.env.scene["robot"].data.root_link_pos_w[0, 0] = 6.1
    self.step()
    self.assertTrue(self.term.clean_finish.item())
    self.term.reset(torch.tensor([0]))
    self.assertEqual(self.env.extras["log"]["Episode_Metrics/clean_descent"].item(), 1.0)
    self.assertEqual(self.term.target.item(), 0)

  def test_brief_touch_does_not_complete(self):
    self.place(0, 0)
    for _ in range(2):
      self.step()
    self.place()
    self.step()
    self.place(0, 0)
    for _ in range(4):
      self.step()
    self.assertEqual(self.term.completed.item(), 0)
    self.step()
    self.assertEqual(self.term.completed.item(), 1)

  def test_later_tread_recovers_without_credit(self):
    self.place(1, 2)
    for _ in range(5):
      reward = self.step()
    self.assertEqual(reward.item(), 0.0)
    self.assertEqual(self.term.target.item(), 3)
    self.assertEqual(self.term.skipped.item(), 2)
    self.assertEqual(self.term.completed.item(), 0)
    self.term.reset(torch.tensor([0]))
    self.assertEqual(self.env.extras["log"]["Episode_Metrics/skipped_treads"].item(), 2.0)

  def test_weak_or_hovering_foot_does_not_complete(self):
    self.place(0, 0)
    self.env.scene["feet_environment_contact"].data.force[0, 0, 2] = -1.0
    for _ in range(6):
      self.step()
    self.assertEqual(self.term.completed.item(), 0)
    self.place(0, 0)
    self.env.scene["left_sole_support"].sole_points_w[..., 2] += 0.08
    for _ in range(6):
      self.step()
    self.assertEqual(self.term.completed.item(), 0)


if __name__ == "__main__":
  unittest.main()
