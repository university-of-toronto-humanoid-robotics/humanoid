"""Check the HoST PPO math (algorithm.py) against HoST's own code, on random data.
No simulator needed. From training/ inside the container:

  python g1_host/scripts/check_ppo.py

Prints PASS/FAIL per check and exits with 1 if anything failed.
"""

import math
import sys

import _setup  # noqa: F401
import torch

from g1_host import params as P
from g1_host.algorithm import HostPPO, MultiCriticActorCritic, MultiCriticStorage
from g1_host.mdp import tolerance

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
  if not ok:
    failures.append(name)


def host_reference_returns(rewards, dones, values, last_values, gamma, lam, weights):
  """HoST rsl_rl/storage/rollout_storage.py:129-146, transcribed (loops and all)."""
  T, N, G = rewards.shape
  returns = torch.zeros_like(rewards)
  advantages = torch.zeros_like(rewards)
  advantage = 0
  for step in reversed(range(T)):
    next_values = last_values if step == T - 1 else values[step + 1]
    nt = 1.0 - dones[step].float()
    delta = rewards[step] + nt * gamma * next_values - values[step]
    advantage = delta + nt * gamma * lam * advantage
    returns[step] = advantage + values[step]
  for c in range(G):
    advantages[:, :, c] = returns[:, :, c] - values[:, :, c]
    advantages[:, :, c] = (advantages[:, :, c] - advantages[:, :, c].mean()) / (advantages[:, :, c].std() + 1e-8)
  return returns, torch.sum(advantages * weights.view(1, 1, -1), dim=-1, keepdim=True)


def host_tolerance(x, bounds, margin, value_at_margin):
  """HoST g1_utils.py:249-266."""
  lower, upper = bounds
  in_bounds = torch.logical_and(lower <= x, x <= upper)
  if margin == 0:
    return torch.where(in_bounds, 1.0, 0.0)
  d = torch.where(x < lower, lower - x, x - upper) / margin
  scale = math.sqrt(-2 * math.log(value_at_margin))
  return torch.where(in_bounds, 1.0, torch.exp(-0.5 * (d * scale) ** 2))


def main() -> None:
  torch.manual_seed(0)
  T, N, G, O, A = 50, 8, 4, 456, 23
  weights = torch.tensor(P.GROUP_WEIGHTS)

  # 1. Multi-critic GAE, per-critic normalization, weighted sum.
  s = MultiCriticStorage(T, N, O, A, G, "cpu")
  s.rewards = torch.randn(T, N, G)
  s.values = torch.randn(T, N, G)
  s.dones = (torch.rand(T, N, 1) < 0.05).float()
  s.dones[0, 0] = 1.0
  s.dones[T - 1, 1] = 1.0
  last = torch.randn(N, G)
  s.compute_returns(last, P.GAMMA, P.LAM, weights)
  ref_ret, ref_adv = host_reference_returns(s.rewards, s.dones, s.values, last, P.GAMMA, P.LAM, weights)
  check("GAE returns match HoST", torch.allclose(s.returns, ref_ret, atol=1e-5))
  check("weighted normalized advantages match HoST", torch.allclose(s.advantages, ref_adv, atol=1e-5),
        f"max diff {float((s.advantages - ref_adv).abs().max()):.2e}")

  # 2. L2C2 coefficients as HoST's code computes them.
  c_pi, c_v = P.l2c2_coefs(P.L2C2_VALUE)
  check("L2C2 policy coefficient = 0.1111", abs(c_pi - 0.1 / 0.9) < 1e-9, f"{c_pi:.5f}")
  check("L2C2 value coefficient = 0.01111", abs(c_v - 0.01 / 0.9) < 1e-9, f"{c_v:.5f}")
  check("L2C2 value coefficient (wall) = 0.02778", abs(P.l2c2_coefs(P.L2C2_VALUE_WALL)[1] - 0.025 / 0.9) < 1e-9)

  # 3. L2C2 losses vanish when the next state equals the current one.
  model = MultiCriticActorCritic(O, A, G)
  obs = torch.randn(64, O)
  mid = obs + (torch.rand(64, 1) * 2 - 1) * (obs - obs)
  l_pi = torch.square(torch.norm(model.act_inference(obs) - model.act_inference(mid), dim=-1)).mean()
  check("L2C2 loss is 0 for s' = s", float(l_pi) == 0.0)

  # 4. Actor output bounded by tanh; 4 critics.
  out = model.act_inference(torch.randn(256, O) * 50)
  check("actor mean in [-1, 1]", bool((out.abs() <= 1.0).all()))
  check("one value per reward group", tuple(model.evaluate(obs).shape) == (64, G))
  check("architecture 512-256-128 / 512-256", [m.out_features for m in model.actor if hasattr(m, "out_features")]
        == [512, 256, 128, A] and [m.out_features for m in model.critics[0] if hasattr(m, "out_features")]
        == [512, 256, 1])
  check("initial action std 0.8", torch.allclose(model.std.detach(), torch.full((A,), 0.8)))

  # 5. tolerance() equals HoST's.
  x = torch.tensor([-1.0, 0.0, 0.5, 0.99, 1.0, 1.5, 3.0])
  for bounds, margin, v in (((1.0, math.inf), 1.0, 0.1), ((0.99, math.inf), 1.0, 0.05), ((0.0, 0.4), 0.0, 0.1)):
    ours = tolerance(x, bounds[0], bounds[1], margin, v)
    check(f"tolerance{bounds, margin, v} matches HoST", torch.allclose(ours, host_tolerance(x, bounds, margin, v)))

  # 6. One full update runs and changes the weights.
  store = MultiCriticStorage(T, N, O, A, G, "cpu")
  alg = HostPPO(model, store, P.GROUP_WEIGHTS, num_mini_batches=4, num_learning_epochs=2)
  before = [p.detach().clone() for p in model.parameters()]
  o = torch.randn(N, O)
  with torch.inference_mode():
    for _ in range(T):
      alg.act(o)
      o = torch.randn(N, O)
      alg.process_env_step(torch.randn(N, G), torch.rand(N) < 0.05, torch.rand(N) < 0.02)
    alg.compute_returns(o)
  losses = alg.update()
  moved = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
  check("an update runs and changes the weights", moved and all(math.isfinite(v) for v in losses.values()),
        ", ".join(f"{k} {v:.3g}" for k, v in losses.items()))

  print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
