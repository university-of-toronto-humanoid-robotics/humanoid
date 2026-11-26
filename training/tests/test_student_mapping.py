"""Deterministic checks of the student elevation-map observation contract."""

import sys
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _setup  # noqa: F401,E402
from g1_comp.rpl import rpl_env_cfg  # noqa: E402
from g1_comp.student_mapping.observation import (  # noqa: E402
  StudentMapCfg, attach_student_map, fill_unknown, select_ground, visibility,
)
from compare_student_ros_scan import ros_projection, ros_scan_module  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402


class StudentMappingTest(unittest.TestCase):
  def test_full_scan_matches_ros_projection(self):
    cfg = attach_student_map(rpl_env_cfg("descend", play=True))
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      env.reset(seed=42)
      term = env.observation_manager.get_term_cfg("actor", "height_scan").func
      self.assertEqual(term.cfg.observation_delay_s, 0.0)
      self.assertGreater((term.last_source == 0).float().mean().item(), 0.0)
      ros = ros_scan_module()
      for step in range(20):  # Cross the 3 Hz map refresh at step 17.
        obs, *_ = env.step(torch.zeros((1, 29)))
        if step in (0, 16, 19):
          expected, observed_share, feet_share = ros_projection(env, term, ros)
          actual = obs["actor"][0, -693:].detach().cpu().numpy()
          np.testing.assert_allclose(actual, expected, atol=1e-6)
          self.assertAlmostEqual((term.last_source == 0).float().mean().item(), observed_share, places=6)
          self.assertAlmostEqual((term.last_source == 1).float().mean().item(), feet_share, places=6)
    finally:
      env.close()

  def test_ros_height_scan_reference(self):
    reference = (Path(__file__).resolve().parents[2] / "ros2_ws/src/perception/height_scan/height_scan/scan.py")
    spec = importlib.util.spec_from_file_location("ros_height_scan_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = np.full((100, 100), np.nan)
    filled = np.full((100, 100), 0.1)
    points = module.scan_points((0., 0.), 0.)
    i = np.floor((2.5 - points[:, 0]) / 0.05).astype(int)
    j = np.floor((2.5 - points[:, 1]) / 0.05).astype(int)
    observed[i[10], j[10]] = 0.2
    expected, _, _ = module.compute_scan(observed, filled, (2.5, 2.5), 0.05,
                                         (0., 0., 0.78), 0., 0.)
    raw = module.sample(observed, (2.5, 2.5), 0.05, points)
    infill = module.sample(filled, (2.5, 2.5), 0.05, points)
    valid = torch.tensor(np.isfinite(raw)).unsqueeze(0)
    fill_valid = torch.tensor(np.isfinite(infill)).unsqueeze(0)
    near = torch.tensor(np.linalg.norm(points, axis=1) < 0.5 - 1e-6).unsqueeze(0)
    ground, _ = select_ground(torch.tensor(np.nan_to_num(raw)).unsqueeze(0), valid,
                              torch.tensor(np.nan_to_num(infill)).unsqueeze(0), fill_valid,
                              near, torch.tensor([[0.]]))
    np.testing.assert_allclose((0.78 - ground.numpy()[0]).clip(-1, 2), expected, atol=1e-6)

  def test_visibility_and_occlusion(self):
    cfg = StudentMapCfg()
    head = torch.tensor([[0., 0., 1.5]])
    hits = torch.tensor([[[0.4, 0., 0.], [2., 0., 0.], [2., 0., 0.], [2., 0., 0.]]])
    intermediate = torch.zeros((1, 4, 6))
    intermediate[0, 2, 3] = 1.0  # wall breaks the line of sight
    valid = torch.tensor([[True, True, True, False]])
    self.assertEqual(visibility(head, hits, valid, cfg, intermediate).tolist(),
                     [[False, True, False, False]])

  def test_inpaint_preserves_observed_and_fallback_order(self):
    observed = torch.tensor([[[0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0.3, 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.]]])
    valid = torch.zeros_like(observed, dtype=torch.bool)
    valid[0, 2, 2] = True
    filled, fill_valid = fill_unknown(observed, valid, 2)
    self.assertTrue(fill_valid.all())
    self.assertAlmostEqual(filled[0, 2, 2].item(), 0.3)
    near = torch.zeros_like(valid)
    near[0, 2, 3] = True
    ground, source = select_ground(observed, valid, filled, fill_valid, near,
                                   torch.tensor([[-0.1]]))
    self.assertEqual(source[0, 2, 2].item(), 0)
    self.assertEqual(source[0, 2, 3].item(), 1)
    self.assertEqual(source[0, 0, 0].item(), 2)
    self.assertAlmostEqual(ground[0, 2, 3].item(), -0.1)

  def test_unseen_cells_use_foot_height(self):
    values = torch.zeros((1, 3))
    valid = torch.zeros_like(values, dtype=torch.bool)
    out, source = select_ground(values, valid, values, valid,
                                torch.zeros_like(valid), torch.tensor([[0.25]]))
    self.assertEqual(source.tolist(), [[1, 1, 1]])
    self.assertTrue(torch.allclose(out, torch.full_like(values, 0.25)))

  def test_four_experts_preserve_other_config(self):
    for feature in ("ascend", "descend", "ramps", "obstacles"):
      with self.subTest(feature=feature):
        original = rpl_env_cfg(feature, play=True)
        student = attach_student_map(original)
        self.assertEqual(list(student.observations["actor"].terms),
                         list(original.observations["actor"].terms))
        self.assertEqual(student.observations["critic"], original.observations["critic"])
        self.assertEqual(student.rewards, original.rewards)
        self.assertEqual(student.actions, original.actions)
        self.assertEqual(student.observations["actor"].terms["height_scan"].params["sensor_name"],
                         "student_map_probe")
        self.assertNotEqual(original.observations["actor"].terms["height_scan"].params["sensor_name"],
                            "student_map_probe")


if __name__ == "__main__":
  unittest.main()
