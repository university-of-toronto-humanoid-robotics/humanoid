"""Contract checks for balanced imitation data and the student observation."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _setup  # noqa: F401,E402

from g1_comp.student_actor import TemporalStudentActor  # noqa: E402
from g1_comp import lanes  # noqa: E402
from g1_comp.rpl import rpl_maze_env_cfg  # noqa: E402
from g1_comp.student_distill import FEATURES, FEATURES_V2, student_env_cfg  # noqa: E402
from g1_comp.student_replay import StudentReplay  # noqa: E402
from g1_comp.student_trainer import (  # noqa: E402
  StudentTrainer, _replay_from_file, _save_atomic, bc_fraction, dagger_label_schedule,
)


class StudentDistillationTest(unittest.TestCase):
  def test_maze_episode_route_stays_in_one_section(self):
    train = rpl_maze_env_cfg()
    play = rpl_maze_env_cfg(play=True)
    self.assertEqual(train.scene.terrain.terrain_generator.size[0], 8.0)
    self.assertEqual(train.scene.terrain.terrain_generator.num_rows, 10)
    self.assertEqual(train.scene.terrain.terrain_generator.num_cols, 1)
    self.assertEqual(play.scene.terrain.terrain_generator.num_rows, 1)
    self.assertEqual(play.scene.terrain.terrain_generator.num_cols, 1)
    self.assertEqual(play.commands["twist"].num_rows, 1)
    self.assertTrue(train.commands["twist"].single_tile_route)
    self.assertEqual(
      train.terminations["rpl_feature_complete"].params["goal_distance"],
      train.metrics["traversal_success"].params["goal_distance"],
    )
    command = lanes.MazeVelocityCommand.__new__(lanes.MazeVelocityCommand)
    command.cfg = train.commands["twist"]
    command.route_x = torch.tensor(lanes.MAZE_ROUTE_X)
    command.route_index = torch.tensor([0, 12, 13, 25])
    self.assertEqual(command._last_route_index().tolist(), [12, 12, 25, 25])

  def test_dagger_samples_across_rollout_without_overwriting_replay(self):
    class Student:
      def __init__(self):
        self.calls = 0

      def __call__(self, obs):
        self.calls += 1
        return torch.zeros((len(obs["student"]), 29))

      def update_normalization(self, obs):
        pass

    class Teacher:
      def __init__(self):
        self.calls = 0

      def __call__(self, obs):
        self.calls += 1
        return obs["student"][:, :1].expand(-1, 29)

    class Env:
      num_envs = 2

      def __init__(self):
        self.step_index = 0
        self.unwrapped = SimpleNamespace(termination_manager=self)

      def step(self, action):
        self.step_index += 1
        values = torch.zeros((2, TemporalStudentActor.OBS_DIM))
        values[:, 0] = self.step_index * 10 + torch.arange(2)
        return TensorDict({"student": values}, batch_size=[2]), None, torch.zeros(2, dtype=torch.bool), None

      def get_term(self, name):
        return torch.zeros(2, dtype=torch.bool)

    env = Env()
    teacher = Teacher()
    student = Student()
    pool = SimpleNamespace(env=env, teacher=teacher,
                           terrain=SimpleNamespace(terrain_levels=torch.zeros(2)),
                           levels=lambda: torch.zeros(2),
                           obs=TensorDict({"student": torch.zeros((2, TemporalStudentActor.OBS_DIM))},
                                          batch_size=[2]))
    trainer = StudentTrainer.__new__(StudentTrainer)
    trainer.stage = "dagger"
    trainer.device = "cpu"
    trainer.args = SimpleNamespace(rollout_steps=6, replay_capacity=40)
    trainer.student = student
    trainer.replay = StudentReplay(40)
    trainer.pools = {"ascend": pool}

    trainer.collect()
    ring = trainer.replay.dagger["ascend"]
    self.assertEqual(student.calls, 6)
    self.assertEqual(teacher.calls, 4)
    self.assertEqual(ring.count, 5)
    values = ring.observations[:ring.count, 0].tolist()
    self.assertEqual(len({int(value) // 10 for value in values}), 4)
    self.assertEqual(ring.actions[:ring.count, 0].tolist(), values)
    trainer.collect()
    self.assertEqual(ring.count, 10)
    self.assertEqual(ring.observations[:5, 0].tolist(), values)

  def test_dagger_sample_budget_is_capped_and_spread_out(self):
    schedule = dagger_label_schedule(24, 6189, 20000)
    self.assertEqual(sum(schedule.values()), 2500)
    self.assertEqual(len(schedule), 4)
    self.assertLessEqual(max(schedule.values()), 6189)
    self.assertTrue(any(step < 6 for step in schedule))
    self.assertTrue(any(step >= 18 for step in schedule))

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
    self.assertEqual(TemporalStudentActor.OBS_DIM, 214 + 5 * 33 * 21)
    self.assertNotIn("terrain_levels", cfg.curriculum)
    dagger = student_env_cfg("ascend", stage="dagger", num_envs=10)
    self.assertIn("terrain_levels", dagger.curriculum)

  def test_five_map_actor_contract(self):
    obs = TensorDict({"student": torch.zeros((2, TemporalStudentActor.OBS_DIM))}, batch_size=[2])
    actor = TemporalStudentActor(obs, {"student": ["student"]}, "student")
    self.assertEqual(actor.terrain_encoder[0].in_channels, 1)
    self.assertEqual(len(actor.map_transformer.layers), 2)
    self.assertEqual(actor.map_transformer.layers[0].self_attn.num_heads, 4)
    self.assertEqual(tuple(actor.map_positions.shape), (1, 5, 128))
    self.assertEqual(actor.map_projection.in_features, 5 * 128)
    self.assertEqual(tuple(actor(obs).shape), (2, 29))

  def test_balanced_replay_and_resume(self):
    replay = StudentReplay(100)
    for terrain_index, feature in enumerate(FEATURES):
      observations = torch.zeros((10, TemporalStudentActor.OBS_DIM))
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

  def test_six_expert_flat_commands_maze_and_balanced_replay(self):
    flat = student_env_cfg("flat", stage="bc", num_envs=60)
    twist = flat.commands["twist"]
    self.assertTrue(twist.heading_command)
    self.assertEqual(twist.ranges.lin_vel_x, (-1.0, 2.0))
    self.assertEqual(twist.ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(twist.ranges.ang_vel_z, (-1.0, 1.0))
    self.assertEqual(flat.scene.terrain.terrain_type, "plane")
    maze = student_env_cfg("maze", stage="dagger", num_envs=60)
    self.assertTrue(maze.scene.terrain.terrain_generator.sub_terrains["maze"].flat_only)
    self.assertIn("terrain_levels", maze.curriculum)
    self.assertIn("rpl_feature_complete", maze.terminations)
    replay = StudentReplay(100, FEATURES_V2)
    for index, feature in enumerate(FEATURES_V2):
      obs = torch.zeros((10, TemporalStudentActor.OBS_DIM))
      obs[:, 0] = index
      labels = torch.zeros((10, 29))
      levels = torch.zeros(10, dtype=torch.long) if feature == "flat" else torch.arange(10)
      replay.add_bc(feature, obs, labels, levels)
      replay.add_dagger(feature, obs + 10, labels)
    bc_obs, _ = replay.sample(60, 1.0, "cpu")
    self.assertEqual(torch.bincount(bc_obs[:, 0].long(), minlength=6).tolist(), [10] * 6)
    mixed, _ = replay.sample(60, 0.5, "cpu")
    self.assertEqual(int((mixed[:, 0] < 10).sum()), 30)
    restored = StudentReplay(100, FEATURES_V2)
    restored.load_state_dict(replay.state_dict())
    self.assertEqual(restored.bc_coverage(), replay.bc_coverage())

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
