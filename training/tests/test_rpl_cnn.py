"""Shape, export, and reward-curriculum checks for the CNN stair task."""

import unittest
from types import SimpleNamespace

import _setup  # noqa: F401
import torch
from tensordict import TensorDict

from g1_comp.rpl_cnn import REWARD_SPEC, cnn_ascend_runner_cfg
from g1_comp.rpl_cnn_actor import RplCnnActor
from g1_comp.rpl_cnn_reward import PenaltyCurriculum, neutral_arm_pose


class RplCnnTests(unittest.TestCase):
  def test_actor_shape_gradients_and_export(self):
    obs = TensorDict({"actor": torch.randn(2, 791)}, batch_size=[2])
    actor = RplCnnActor(obs, {"policy": ["actor"]}, "policy", 29,
                        hidden_dims=(512, 256, 256, 128), obs_normalization=True)
    self.assertEqual(actor.terrain_encoder[:4](obs["actor"][:, 98:].reshape(2, 1, 33, 21)).shape,
                     (2, 16, 8, 5))
    output = actor(obs)
    self.assertEqual(output.shape, (2, 29))
    output.sum().backward()
    self.assertIsNotNone(actor.terrain_encoder[0].weight.grad)
    actor.update_normalization(obs)
    exported = actor.as_onnx()
    self.assertEqual(exported.get_dummy_inputs()[0].shape, (1, 791))
    torch.testing.assert_close(exported(obs["actor"]), actor(obs))
    torch.testing.assert_close(torch.jit.script(actor.as_jit())(obs["actor"]), actor(obs))

  def test_reward_config_and_neutral_pose(self):
    names = {name for name, *_ in REWARD_SPEC}
    self.assertEqual(len(names), len(REWARD_SPEC))
    weights = {name: weight for name, _, weight, _, _ in REWARD_SPEC}
    self.assertEqual(weights["penalty_stumble"], -50.0)
    self.assertEqual(weights["penalty_contact_pattern"], 0.0)
    self.assertEqual(cnn_ascend_runner_cfg().actor.hidden_dims, (512, 256, 256, 128))
    neutral = torch.tensor([[0.35, 0.87, -0.35, -0.87]])
    data = SimpleNamespace(joint_pos=neutral.clone(), default_joint_pos=neutral)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    cfg = SimpleNamespace(joint_ids=[0, 1, 2, 3])
    self.assertEqual(neutral_arm_pose(env, cfg).item(), 0.0)
    data.joint_pos[0, 0] += 0.5
    self.assertAlmostEqual(neutral_arm_pose(env, cfg).item(), 0.25)

  def test_penalty_curriculum_uses_episode_length(self):
    env = SimpleNamespace(device="cpu", episode_length_buf=torch.tensor([220, 220]))
    curriculum = PenaltyCurriculum()
    ids = torch.tensor([0, 1])
    self.assertAlmostEqual(curriculum(env, ids).item(), 0.1 * (1 + 1e-5))
    env.episode_length_buf[:] = 20
    self.assertAlmostEqual(curriculum(env, ids).item(), 0.1 * (1 + 1e-5) * (1 - 1e-5))


if __name__ == "__main__":
  unittest.main()
