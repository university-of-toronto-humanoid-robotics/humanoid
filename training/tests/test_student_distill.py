"""Contract checks for balanced imitation data and the student observation."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _setup  # noqa: F401,E402

from g1_comp.student_actor import TemporalStudentActor  # noqa: E402
from g1_comp.student_distill import FEATURES, student_env_cfg  # noqa: E402
from g1_comp.student_replay import StudentReplay  # noqa: E402
from g1_comp.student_trainer import _replay_from_file, _save_atomic, bc_fraction  # noqa: E402


class StudentDistillationTest(unittest.TestCase):
  def test_observation_groups_keep_expert_clean_and_student_temporal(self):
    cfg = student_env_cfg("ascend", stage="bc", num_envs=10)
    teacher = cfg.observations["actor"]
    student = cfg.observations["student"]
    self.assertFalse(teacher.enable_corruption)
    self.assertTrue(student.enable_corruption)
    self.assertEqual(student.terms["joint_pos"].history_length, 5)
    self.assertEqual(student.terms["joint_pos"].noise.std, 0.01)
    self.assertEqual(student.terms["joint_vel"].noise.std, 0.5)
    self.assertTrue(student.terms["height_scan"].params["return_history"])
    self.assertEqual(TemporalStudentActor.OBS_DIM, 1600)
    self.assertNotIn("terrain_levels", cfg.curriculum)
    dagger = student_env_cfg("ascend", stage="dagger", num_envs=10)
    self.assertIn("terrain_levels", dagger.curriculum)

  def test_balanced_replay_and_resume(self):
    replay = StudentReplay(100)
    for terrain_index, feature in enumerate(FEATURES):
      observations = torch.zeros((10, 1600))
      observations[:, 0] = torch.arange(10) + terrain_index * 10
      labels = torch.zeros((10, 29))
      levels = torch.arange(10)
      replay.add_bc(feature, observations, labels, levels)
      replay.add_dagger(feature, observations + 100, labels)
    bc_obs, _ = replay.sample(40, 1.0, "cpu")
    self.assertEqual(sorted(bc_obs[:, 0].int().tolist()), list(range(40)))
    mixed, _ = replay.sample(80, 0.5, "cpu")
    self.assertEqual(int((mixed[:, 0] < 100).sum()), 40)
    self.assertEqual(int((mixed[:, 0] >= 100).sum()), 40)
    restored = StudentReplay(100)
    restored.load_state_dict(replay.state_dict())
    self.assertEqual(restored.bc_coverage(), replay.bc_coverage())
    self.assertEqual([len(restored.dagger[f]) for f in FEATURES], [10] * 4)
    smaller = StudentReplay(20)
    smaller.load_bc_demos(replay.state_dict())
    self.assertEqual(smaller.bc_coverage(), {feature: [1] * 10 for feature in FEATURES})
    self.assertAlmostEqual(bc_fraction(0), 0.5)
    self.assertAlmostEqual(bc_fraction(1000), 0.1)

  def test_separate_demo_file_and_legacy_checkpoint(self):
    with tempfile.TemporaryDirectory() as directory:
      weights = Path(directory) / "student_10.pt"
      demos = weights.with_name("demos_10.pt")
      metadata = {"stage": "bc", "update": 10, "teacher_hashes": {"ascend": "hash"}}
      _save_atomic(demos, {**metadata, "replay": {"samples": torch.arange(3)}})
      _save_atomic(weights, {**metadata, "demo_file": demos.name, "student": {}})
      state = torch.load(weights, weights_only=False)
      self.assertNotIn("replay", state)
      torch.testing.assert_close(_replay_from_file(weights, state)["samples"], torch.arange(3))
      legacy = {**metadata, "replay": {"samples": torch.arange(2)}}
      torch.testing.assert_close(_replay_from_file(weights, legacy)["samples"], torch.arange(2))
      _save_atomic(demos, {**metadata, "update": 9, "replay": {}})
      with self.assertRaisesRegex(ValueError, "does not match"):
        _replay_from_file(weights, state)


if __name__ == "__main__":
  unittest.main()
