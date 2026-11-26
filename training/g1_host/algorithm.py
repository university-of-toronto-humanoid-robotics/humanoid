"""HoST's PPO: one actor, one critic per reward group, and L2C2 smoothness.

Ported from HoST's fork of rsl_rl (rsl_rl/modules/actor_critic.py,
storage/rollout_storage.py, algorithms/ppo.py; BSD-3, (c) ETH Zurich and NVIDIA;
HoST changes MIT, (c) the HoST authors) to plain torch, because the rsl_rl we
use (5.0.1) builds one single-output critic.

  Actor     MLP 512-256-128, ELU, tanh on the output, learnable std (init 0.8).
  Critics   one MLP 512-256 (ELU) per reward group (task, regu, style, post),
            each with its own value target.
  Advantage GAE per group; each group's advantages normalized over the batch, then
            A = sum_g w_g * A_g with w = (2.5, 0.1, 1, 1)  (rollout_storage.py:129-146).
  L2C2      on a state between s_t and s_t+1, s = s_t + u (s_t+1 - s_t), u ~ U(-1, 1),
            zero where the episode ended: loss += c_pi ||mu(s_t) - mu(s)||^2
            + c_V ||V(s_t) - V(s)||^2 over all critics  (ppo.py:182-195).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(inp: int, hidden: tuple[int, ...], out: int, tanh_out: bool = False) -> nn.Sequential:
  layers: list[nn.Module] = []
  dims = (inp, *hidden)
  for a, b in zip(dims[:-1], dims[1:]):
    layers += [nn.Linear(a, b), nn.ELU()]
  layers.append(nn.Linear(dims[-1], out))
  if tanh_out:
    layers.append(nn.Tanh())
  return nn.Sequential(*layers)


class MultiCriticActorCritic(nn.Module):
  def __init__(self, num_obs: int, num_actions: int, num_critics: int, actor_hidden=(512, 256, 128),
               critic_hidden=(512, 256), init_std: float = 0.8):
    super().__init__()
    self.actor = _mlp(num_obs, tuple(actor_hidden), num_actions, tanh_out=True)
    self.critics = nn.ModuleList(_mlp(num_obs, tuple(critic_hidden), 1) for _ in range(num_critics))
    self.std = nn.Parameter(init_std * torch.ones(num_actions))
    Normal.set_default_validate_args(False)

  def distribution(self, obs: torch.Tensor) -> Normal:
    mean = self.actor(obs)
    return Normal(mean, mean * 0.0 + self.std)

  def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(obs)

  def evaluate(self, obs: torch.Tensor) -> torch.Tensor:  # [B, G]
    return torch.cat([c(obs) for c in self.critics], dim=-1)


class MultiCriticStorage:
  def __init__(self, num_steps: int, num_envs: int, num_obs: int, num_actions: int, num_critics: int, device):
    t, n, g = num_steps, num_envs, num_critics
    z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
    self.obs = z(t, n, num_obs)
    self.actions = z(t, n, num_actions)
    self.rewards = z(t, n, g)
    self.dones = z(t, n, 1)
    self.values = z(t, n, g)
    self.returns = z(t, n, g)
    self.advantages = z(t, n, 1)
    self.logp = z(t, n, 1)
    self.mu = z(t, n, num_actions)
    self.sigma = z(t, n, num_actions)
    self.num_steps, self.num_envs, self.step = t, n, 0

  def add(self, obs, actions, rewards, dones, values, logp, mu, sigma) -> None:
    i = self.step
    self.obs[i].copy_(obs)
    self.actions[i].copy_(actions)
    self.rewards[i].copy_(rewards)
    self.dones[i].copy_(dones.view(-1, 1).float())
    self.values[i].copy_(values)
    self.logp[i].copy_(logp.view(-1, 1))
    self.mu[i].copy_(mu)
    self.sigma[i].copy_(sigma)
    self.step += 1

  def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float, group_weights: torch.Tensor) -> None:
    adv = torch.zeros_like(last_values)
    for t in reversed(range(self.num_steps)):
      next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
      not_done = 1.0 - self.dones[t]
      delta = self.rewards[t] + not_done * gamma * next_values - self.values[t]
      adv = delta + not_done * gamma * lam * adv
      self.returns[t] = adv + self.values[t]
    a = self.returns - self.values
    a = (a - a.mean(dim=(0, 1))) / (a.std(dim=(0, 1)) + 1e-8)  # each critic over the whole batch
    self.advantages = (a * group_weights).sum(dim=-1, keepdim=True)

  def mini_batches(self, num_mini_batches: int, num_epochs: int):
    """As HoST: steps 0..T-2, each with its next observation (for L2C2)."""
    n = self.num_envs * (self.num_steps - 1)
    size = n // num_mini_batches
    flat = lambda x: x[:-1].flatten(0, 1)  # noqa: E731
    obs, next_obs = flat(self.obs), self.obs[1:].flatten(0, 1)
    cont = 1.0 - flat(self.dones)
    data = (obs, next_obs, cont, flat(self.actions), flat(self.values), flat(self.advantages), flat(self.returns),
            flat(self.logp), flat(self.mu), flat(self.sigma))
    for _ in range(num_epochs):
      perm = torch.randperm(num_mini_batches * size, device=obs.device)
      for i in range(num_mini_batches):
        idx = perm[i * size:(i + 1) * size]
        yield tuple(x[idx] for x in data)


class HostPPO:
  def __init__(self, model: MultiCriticActorCritic, storage: MultiCriticStorage, group_weights, *,
               num_learning_epochs=5, num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0,
               entropy_coef=0.01, learning_rate=1e-3, max_grad_norm=1.0, use_clipped_value_loss=True,
               schedule="adaptive", desired_kl=0.01, l2c2_policy=0.111, l2c2_value=0.0111, device="cpu"):
    self.model, self.storage, self.device = model, storage, device
    self.group_weights = torch.tensor(group_weights, device=device, dtype=torch.float)
    self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    self.epochs, self.mini_batches = num_learning_epochs, num_mini_batches
    self.clip, self.gamma, self.lam = clip_param, gamma, lam
    self.value_coef, self.entropy_coef = value_loss_coef, entropy_coef
    self.learning_rate, self.max_grad_norm = learning_rate, max_grad_norm
    self.clipped_value, self.schedule, self.desired_kl = use_clipped_value_loss, schedule, desired_kl
    self.l2c2_policy, self.l2c2_value = l2c2_policy, l2c2_value
    self._pending: tuple | None = None

  # --- rollout ---------------------------------------------------------------------

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    dist = self.model.distribution(obs)
    actions = dist.sample()
    self._pending = (obs, actions, self.model.evaluate(obs), dist.log_prob(actions).sum(-1), dist.mean, dist.stddev)
    return actions

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, time_outs: torch.Tensor | None) -> None:
    obs, actions, values, logp, mu, sigma = self._pending
    rewards = rewards.clone()
    if time_outs is not None:  # bootstrap the value of an episode cut by the time limit, per critic
      rewards += self.gamma * values * time_outs.view(-1, 1).float()
    self.storage.add(obs, actions, rewards, dones, values, logp, mu, sigma)

  def compute_returns(self, last_obs: torch.Tensor) -> None:
    self.storage.compute_returns(self.model.evaluate(last_obs).detach(), self.gamma, self.lam, self.group_weights)

  # --- update -----------------------------------------------------------------------

  def update(self) -> dict[str, float]:
    sums = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "l2c2_policy": 0.0, "l2c2_value": 0.0, "kl": 0.0,
            "action_smoothness": 0.0}
    count = 0
    for obs, next_obs, cont, actions, old_values, adv, returns, old_logp, old_mu, old_sigma in \
        self.storage.mini_batches(self.mini_batches, self.epochs):
      dist = self.model.distribution(obs)
      logp = dist.log_prob(actions).sum(-1)
      values = self.model.evaluate(obs)
      mu, sigma = dist.mean, dist.stddev
      entropy = dist.entropy().sum(-1)

      with torch.no_grad():
        kl = torch.sum(torch.log(sigma / old_sigma + 1e-5)
                       + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square()) - 0.5, dim=-1).mean()
      if self.desired_kl is not None and self.schedule == "adaptive":
        if kl > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
          group["lr"] = self.learning_rate

      ratio = torch.exp(logp - old_logp.squeeze(-1))
      a = adv.squeeze(-1)
      surrogate = torch.max(-a * ratio, -a * ratio.clamp(1.0 - self.clip, 1.0 + self.clip)).mean()
      if self.clipped_value:
        clipped = old_values + (values - old_values).clamp(-self.clip, self.clip)
        value_loss = torch.max((values - returns).square(), (clipped - returns).square()).mean()
      else:
        value_loss = (returns - values).square().mean()

      # L2C2 (ppo.py:182-195)
      u = cont * (torch.rand_like(cont) - 0.5) * 2.0
      mid = obs + u * (next_obs - obs)
      l2c2_pi = torch.square(torch.norm(mu - self.model.act_inference(mid), dim=-1)).mean()
      l2c2_v = torch.square(torch.norm(values - self.model.evaluate(mid), dim=-1)).mean()

      loss = (surrogate + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
              + self.l2c2_policy * l2c2_pi + self.l2c2_value * l2c2_v)
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
      self.optimizer.step()

      with torch.no_grad():
        smooth = torch.norm(mu - self.model.act_inference(next_obs), dim=-1).mean()
      for k, v in (("value", value_loss), ("surrogate", surrogate), ("entropy", entropy.mean()),
                   ("l2c2_policy", l2c2_pi), ("l2c2_value", l2c2_v), ("kl", kl), ("action_smoothness", smooth)):
        sums[k] += float(v)
      count += 1
    self.storage.step = 0
    return {k: v / max(count, 1) for k, v in sums.items()}
