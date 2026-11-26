"""The runner train.py, play.py and the video/eval scripts use for the HoST tasks.

Same interface as mjlab's runner (constructor, learn, save, load,
get_inference_policy), with HoST's multi-critic PPO (algorithm.py) inside. The
reward of each group comes from mjlab's per-term rewards of the step
(reward_manager._step_reward = value x weight, before dt):

  task  = product of the task_ terms          (not x dt, as HoST)
  regu, style, post = sum of their terms x dt

A checkpoint (model_<N>.pt) also keeps every robot's pull force and action bound,
so a run that is stopped and resumed (24 h cluster jobs) continues the curriculum
where it was. Next to it: policy.onnx (the actor, 456 inputs -> 23 actions in
[-1, 1]) and policy_meta.json (everything the robot side needs to run it).
"""

from __future__ import annotations

import copy
import json
import os
import statistics
import subprocess
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from . import params as P
from . import robot
from .algorithm import HostPPO, MultiCriticActorCritic, MultiCriticStorage
from .tasks import HOST_ALGO


def _step_rewards(base) -> torch.Tensor:
  try:
    return base.reward_manager._step_reward
  except AttributeError as e:  # mjlab is pinned to 1.2.0; a newer one may rename it
    raise SystemExit("mjlab's RewardManager has no _step_reward any more: update g1_host/runner.py") from e


class HostRunner:
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    self.env, self.cfg, self.device, self.log_dir = env, train_cfg, device, log_dir
    base = env.unwrapped
    extra = HOST_ALGO.get(train_cfg.get("experiment_name", ""), {})
    self.task = extra.get("task", train_cfg.get("experiment_name", "?"))
    group_weights = extra.get("group_weights", P.GROUP_WEIGHTS)
    l2c2_value = extra.get("l2c2_value", P.L2C2_VALUE)

    names = list(base.reward_manager.active_terms)
    self.groups = [torch.tensor([i for i, n in enumerate(names) if n.startswith(g + "_")], dtype=torch.long,
                                device=device) for g in P.GROUPS]
    known = {i for t in self.groups for i in t.tolist()}
    unknown = [n for i, n in enumerate(names) if i not in known]
    if unknown:
      raise SystemExit(f"reward terms without a group prefix ({'/'.join(P.GROUPS)}_): {unknown}")

    obs = env.get_observations()["actor"]
    self.num_obs, self.num_actions = obs.shape[1], env.num_actions
    actor, critic, alg = train_cfg["actor"], train_cfg["critic"], train_cfg["algorithm"]
    init_std = (actor.get("distribution_cfg") or {}).get("init_std", P.INIT_NOISE_STD)
    self.model = MultiCriticActorCritic(self.num_obs, self.num_actions, len(P.GROUPS), tuple(actor["hidden_dims"]),
                                        tuple(critic["hidden_dims"]), init_std).to(device)
    self.num_steps = train_cfg["num_steps_per_env"]
    storage = MultiCriticStorage(self.num_steps, env.num_envs, self.num_obs, self.num_actions, len(P.GROUPS), device)
    c_pi, c_v = P.l2c2_coefs(l2c2_value)
    self.alg = HostPPO(
      self.model, storage, group_weights, num_learning_epochs=alg["num_learning_epochs"],
      num_mini_batches=alg["num_mini_batches"], clip_param=alg["clip_param"], gamma=alg["gamma"], lam=alg["lam"],
      value_loss_coef=alg["value_loss_coef"], entropy_coef=alg["entropy_coef"], learning_rate=alg["learning_rate"],
      max_grad_norm=alg["max_grad_norm"], use_clipped_value_loss=alg["use_clipped_value_loss"],
      schedule=alg["schedule"], desired_kl=alg["desired_kl"], l2c2_policy=c_pi, l2c2_value=c_v, device=device,
    )
    self.current_learning_iteration = 0
    self.writer: SummaryWriter | None = None
    self._git_paths: list[str] = []

  # --- rewards ---------------------------------------------------------------------

  def group_rewards(self) -> torch.Tensor:
    r = _step_rewards(self.env.unwrapped)
    dt = self.env.unwrapped.step_dt
    out = []
    for g, cols in zip(P.GROUPS, self.groups):
      if len(cols) == 0:
        out.append(torch.zeros(r.shape[0], device=r.device))
      elif g == "task":
        out.append(torch.prod(r[:, cols], dim=1))
      else:
        out.append(r[:, cols].sum(dim=1) * dt)
    return torch.stack(out, dim=1).to(self.device)

  # --- training ------------------------------------------------------------------------

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    env = self.env
    if self.log_dir is not None and self.writer is None:
      os.makedirs(self.log_dir, exist_ok=True)
      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
      self._write_git_info()
    if init_at_random_ep_len:
      env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = env.get_observations()["actor"].to(self.device)
    self.model.train()
    n = env.num_envs
    group_sum = torch.zeros(n, len(P.GROUPS), device=self.device)
    ep_len = torch.zeros(n, device=self.device)
    returns, lengths = deque(maxlen=200), deque(maxlen=200)
    group_returns = [deque(maxlen=200) for _ in P.GROUPS]
    start, total = self.current_learning_iteration, self.current_learning_iteration + num_learning_iterations
    t_start = time.time()
    for it in range(start, total):
      t0 = time.time()
      logs: dict[str, list[float]] = {}
      with torch.inference_mode():
        for _ in range(self.num_steps):
          actions = self.alg.act(obs)
          obs_td, _, dones, extras = env.step(actions.to(env.device))
          rewards = self.group_rewards()
          obs = obs_td["actor"].to(self.device)
          dones = dones.to(self.device)
          self.alg.process_env_step(rewards, dones, extras.get("time_outs"))
          group_sum += rewards
          ep_len += 1
          done_ids = dones.nonzero().flatten()
          if len(done_ids):
            for g in range(len(P.GROUPS)):
              group_returns[g].extend(group_sum[done_ids, g].tolist())
            returns.extend((group_sum[done_ids] * self.alg.group_weights).sum(dim=1).tolist())
            lengths.extend(ep_len[done_ids].tolist())
            group_sum[done_ids] = 0.0
            ep_len[done_ids] = 0.0
          for k, v in extras.get("log", {}).items():
            logs.setdefault(k, []).append(float(v))
        t1 = time.time()
        self.alg.compute_returns(obs)
      losses = self.alg.update()
      t2 = time.time()
      self.current_learning_iteration = it
      self._log(it, total, t1 - t0, t2 - t1, losses, logs, returns, lengths, group_returns, time.time() - t_start,
                start)
      if self.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
    if self.writer is not None:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.writer.close()

  def _log(self, it, total, collect, learn, losses, logs, returns, lengths, group_returns, elapsed, start) -> None:
    steps = self.num_steps * self.env.num_envs
    fps = int(steps / max(collect + learn, 1e-6))
    std = float(self.model.std.mean())
    w = self.writer
    if w is not None:
      for k, v in losses.items():
        w.add_scalar(f"Loss/{k}", v, it)
      w.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
      w.add_scalar("Policy/mean_noise_std", std, it)
      w.add_scalar("Perf/total_fps", fps, it)
      w.add_scalar("Perf/collection_time", collect, it)
      w.add_scalar("Perf/learning_time", learn, it)
      if returns:
        w.add_scalar("Train/mean_reward", statistics.mean(returns), it)
        w.add_scalar("Train/mean_episode_length", statistics.mean(lengths), it)
        for g, buf in zip(P.GROUPS, group_returns):
          w.add_scalar(f"Reward_group/{g}", statistics.mean(buf), it)
      for k, v in logs.items():
        w.add_scalar(k, sum(v) / len(v), it)
    lines = [f"{'#' * 80}", f"Learning iteration {it}/{total}".center(80), "",
             f"{'Computation:':>35} {fps} steps/s (collection: {collect:.3f}s, learning {learn:.3f}s)"]
    lines += [f"{k + ' loss:':>35} {v:.4f}" for k, v in losses.items()]
    lines.append(f"{'Mean action noise std:':>35} {std:.2f}")
    if returns:
      lines.append(f"{'Mean reward (weighted groups):':>35} {statistics.mean(returns):.2f}")
      lines.append(f"{'Mean episode length:':>35} {statistics.mean(lengths):.2f}")
      lines += [f"{'Group ' + g + ':':>35} {statistics.mean(b):.3f}" for g, b in zip(P.GROUPS, group_returns)]
    for k in sorted(logs):
      if k.startswith("Curriculum/"):
        lines.append(f"{k + ':':>35} {sum(logs[k]) / len(logs[k]):.3f}")
    done = it - start + 1
    lines += [f"{'Iteration time:':>35} {collect + learn:.2f}s",
              f"{'ETA:':>35} {elapsed / done * (total - it - 1):.0f}s"]
    print("\n".join(lines), flush=True)

  # --- save / load / export -------------------------------------------------------------

  def _curriculum_terms(self):
    act = self.env.unwrapped.action_manager.get_term("joint_pos")
    return act.pull_force, act.beta

  def save(self, path: str, infos: dict | None = None) -> None:
    force, beta = self._curriculum_terms()
    base = self.env.unwrapped
    torch.save({
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "learning_rate": self.alg.learning_rate,
      "infos": {**(infos or {}), "env_state": {"common_step_counter": base.common_step_counter},
                "host_curriculum": {"pull_force": force.cpu(), "beta": beta.cpu()}},
      "host": {"task": self.task, "num_obs": self.num_obs, "num_actions": self.num_actions},
    }, path)
    self.export(os.path.dirname(path))

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
    d = torch.load(path, map_location=map_location or self.device, weights_only=False)
    if "model_state_dict" not in d or not any(k.startswith("critics.") for k in d["model_state_dict"]):
      raise SystemExit(f"{path} is not a HoST checkpoint (no multi-critic model inside)")
    actor_only = load_cfg is not None and {k for k, v in load_cfg.items() if v} == {"actor"}
    state = d["model_state_dict"]
    if actor_only:
      self.model.load_state_dict({k: v for k, v in state.items() if k.startswith("actor.") or k == "std"}, strict=False)
      return d.get("infos") or {}
    self.model.load_state_dict(state, strict=strict)
    if "optimizer_state_dict" in d:
      self.alg.optimizer.load_state_dict(d["optimizer_state_dict"])
    self.alg.learning_rate = d.get("learning_rate", self.alg.learning_rate)
    self.current_learning_iteration = d.get("iter", -1) + 1  # model_<N>.pt is saved after iteration N
    infos = d.get("infos") or {}
    cur = infos.get("host_curriculum")
    if cur:
      force, beta = self._curriculum_terms()
      for dst, src in ((force, cur["pull_force"]), (beta, cur["beta"])):
        src = src.to(dst.device)
        dst[:] = src if src.shape == dst.shape else src.mean()
    if infos.get("env_state"):
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def get_inference_policy(self, device: str | None = None):
    self.model.eval()
    if device is not None:
      self.model.to(device)
    model = self.model

    def policy(obs):
      return model.act_inference(obs if isinstance(obs, torch.Tensor) else obs["actor"])

    return policy

  def export(self, path: str) -> None:
    """policy.onnx and policy_meta.json in `path`."""
    actor = copy.deepcopy(self.model.actor).to("cpu").eval()
    try:
      torch.onnx.export(actor, torch.zeros(1, self.num_obs), os.path.join(path, "policy.onnx"), input_names=["obs"],
                        output_names=["actions"], opset_version=18, dynamo=False)
    except Exception as e:  # never lose a checkpoint over the export
      print(f"[WARN] ONNX export failed: {e}")
    joints = list(robot.policy_joints())
    meta = {
      "task": self.task,
      "iteration": self.current_learning_iteration,
      "control_hz": 1.0 / (P.SIM_DT * P.DECIMATION),
      "joint_names": joints,
      "held_joints_at_zero": list(robot.held_joints()),
      "kp": [robot.gains(j)[0] for j in joints],
      "kd": [robot.gains(j)[1] for j in joints],
      "action_law": "q_target = q_measured + beta * action, sent once per policy step (paper Eq. 1)",
      "beta": P.BETA_EVAL,
      "imu": "pelvis",
      "observation": {
        "history": P.HISTORY,
        "order": "oldest frame first",
        "frame": ["base_ang_vel x0.25 (3)", "projected_gravity (3)", "joint_pos absolute (23)",
                  "joint_vel x0.05 (23)", "last action (23)", "beta (1)"],
        "frame_size": self.num_obs // P.HISTORY,
      },
      "notes": "The paper's hardware deployment raised hip/knee Kp to 200/275 and used beta 0.3.",
    }
    with open(os.path.join(path, "policy_meta.json"), "w") as f:
      json.dump(meta, f, indent=2)

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    self._git_paths.append(repo_file_path)

  def _write_git_info(self) -> None:
    if not self._git_paths or self.log_dir is None:
      return
    try:
      where = os.path.dirname(os.path.abspath(__file__))
      head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=where, capture_output=True, text=True).stdout.strip()
      diff = subprocess.run(["git", "diff", "--stat"], cwd=where, capture_output=True, text=True).stdout
      os.makedirs(os.path.join(self.log_dir, "params"), exist_ok=True)
      with open(os.path.join(self.log_dir, "params", "git.txt"), "w") as f:
        f.write(f"{head}\n{diff}")
    except OSError:
      pass
