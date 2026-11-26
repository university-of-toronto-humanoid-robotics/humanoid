"""Shared rollout and optimization engine for BC warmup and DAgger."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
import os
import random

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from .student_actor import TemporalStudentActor
from .student_distill import FEATURES, TASKS, student_env_cfg
from .student_mapping.observation import StudentMapCfg
from .student_replay import StudentReplay


def bc_fraction(update: int, decay_updates: int = 1000) -> float:
  return 0.5 - 0.4 * min(update / decay_updates, 1.0)


def add_common_args(parser: argparse.ArgumentParser):
  parser.add_argument("--teacher-ascend", type=Path, required=True)
  parser.add_argument("--teacher-descend", type=Path, required=True)
  parser.add_argument("--teacher-ramps", type=Path, required=True)
  parser.add_argument("--teacher-obstacles", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=1024, help="Total across four terrains")
  parser.add_argument("--rollout-steps", type=int, default=24)
  parser.add_argument("--updates", type=int, required=True)
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--gradient-steps", type=int, default=4)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--replay-capacity", type=int, default=50000, help="Per terrain, per source")
  parser.add_argument("--save-every", type=int, default=100)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--run-name", default="student")
  parser.add_argument("--run-dir", type=Path)
  parser.add_argument("--logger", choices=("tensorboard", "wandb", "none"), default="tensorboard")
  parser.add_argument("--wandb-project", default="humanoid-student")
  parser.add_argument("--resume", type=Path, help="Full checkpoint of the same stage")


def validate_args(args, stage):
  if args.num_envs < (40 if stage == "bc" else 4) or args.num_envs % 4:
    raise ValueError("BC needs at least 40 total envs; both stages need a multiple of four")
  for name in ("rollout_steps", "updates", "batch_size", "gradient_steps", "save_every", "replay_capacity"):
    if getattr(args, name) < 1:
      raise ValueError(f"{name} must be positive")
  if args.learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
  for feature in FEATURES:
    path = getattr(args, f"teacher_{feature}")
    if not path.is_file():
      raise FileNotFoundError(f"Missing {feature} expert: {path}")
  if stage == "dagger":
    if args.resume and (args.init_mode or args.bc_checkpoint or args.bc_demo_checkpoint):
      raise ValueError("--resume cannot be combined with a DAgger start mode")
    if not args.resume:
      args.init_mode = args.init_mode or "scratch"
      if args.init_mode != "scratch" and not args.bc_checkpoint:
        raise ValueError("BC initialization requires --bc-checkpoint")
      if args.init_mode == "scratch" and args.bc_checkpoint:
        raise ValueError("Scratch mode cannot take a BC checkpoint")
      if args.init_mode != "bc-with-demos" and args.bc_demo_checkpoint:
        raise ValueError("--bc-demo-checkpoint is only for bc-with-demos")


def _rng_state():
  return {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state(),
          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if state["cuda"] is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["cuda"])


def _file_sha256(path: str) -> str:
  with open(path, "rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def _replay_from_file(path: Path, state: dict) -> dict:
  """Read a separate replay file, or a replay embedded in an older checkpoint."""
  if "replay" in state:
    return state["replay"]
  demo_path = path.with_name(state.get("demo_file", path.name.replace("student_", "demos_", 1)))
  demos = torch.load(demo_path, map_location="cpu", weights_only=False)
  if (demos.get("stage") != state.get("stage") or
      demos.get("update") != state.get("update") or
      demos.get("teacher_hashes") != state.get("teacher_hashes")):
    raise ValueError(f"Demo file does not match its checkpoint: {demo_path}")
  return demos["replay"]


def _save_atomic(path: Path, state: dict):
  temporary = path.with_suffix(".tmp")
  torch.save(state, temporary)
  os.replace(temporary, path)


class Pool:
  def __init__(self, feature, num_envs, device, seed, stage, teacher_checkpoint, levels=None):
    cfg = student_env_cfg(feature, stage=stage, num_envs=num_envs, resume_levels=levels)
    cfg.seed = seed
    self.feature = feature
    agent_cfg = load_rl_cfg(TASKS[feature])
    self.env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                                 clip_actions=agent_cfg.clip_actions)
    try:
      self.obs = self.env.get_observations()
      if self.obs["actor"].shape[-1] != 791 or self.obs["student"].shape[-1] != 1600:
        raise ValueError(f"Unexpected teacher/student observation layout for {feature}")
      runner = load_runner_cls(TASKS[feature])(self.env, asdict(agent_cfg), device=device)
      runner.load(str(teacher_checkpoint), load_cfg={"actor": True}, strict=True,
                  map_location=device)
      self.teacher = runner.get_inference_policy(device=device).eval()
      self.teacher.requires_grad_(False)
    except Exception:
      self.env.close()
      raise

  @property
  def terrain(self):
    return self.env.unwrapped.scene.terrain

  def close(self):
    self.env.close()


class StudentTrainer:
  def __init__(self, args, stage):
    validate_args(args, stage)
    self.args, self.stage = args, stage
    self.device = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(args.seed)
    self.source_paths = {f: str(getattr(args, f"teacher_{f}").resolve()) for f in FEATURES}
    self.teacher_hashes = {f: _file_sha256(path) for f, path in self.source_paths.items()}
    self.saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    self.saved_replay = _replay_from_file(args.resume, self.saved) if self.saved else None
    if self.saved:
      if self.saved["stage"] != stage or self.saved["teacher_hashes"] != self.teacher_hashes:
        raise ValueError("Resume checkpoint stage or expert contents differ")
      if stage == "dagger":
        args.init_mode = self.saved.get("init_mode") or (
          "bc-with-demos" if any(any(bucket["count"] for bucket in buckets)
                                 for buckets in self.saved_replay["bc"].values()) else "scratch"
        )
      if self.saved["num_envs"] != args.num_envs:
        raise ValueError("Full resume requires the original --num-envs")
      if self.saved_replay["capacity"] != args.replay_capacity:
        raise ValueError("Full resume requires the original --replay-capacity")
      if self.saved.get("learning_rate", args.learning_rate) != args.learning_rate:
        raise ValueError("Full resume requires the original --learning-rate")
      if self.saved.get("map_config", asdict(StudentMapCfg())) != asdict(StudentMapCfg()):
        raise ValueError("Student map settings changed since this checkpoint")
    self.run_dir = args.run_dir or (args.resume.parent.parent if args.resume else
                   Path(__file__).resolve().parents[1] / "logs/student_distill" / stage /
                   (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name))
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.pools = {}
    try:
      for index, feature in enumerate(FEATURES):
        levels = self.saved["levels"][feature] if self.saved and stage == "dagger" else None
        self.pools[feature] = Pool(feature, args.num_envs // 4, args.device,
                                   args.seed + index, stage, self.source_paths[feature], levels)
    except Exception:
      self.close()
      raise
    obs = self.pools[FEATURES[0]].obs
    self.student = TemporalStudentActor(obs, {"student": ["student"]}, "student").to(args.device)
    self.optimizer = torch.optim.Adam(self.student.parameters(), lr=args.learning_rate)
    self.replay = StudentReplay(args.replay_capacity)
    self.start_update = 0
    if self.saved:
      self.student.load_state_dict(self.saved["student"])
      self.optimizer.load_state_dict(self.saved["optimizer"])
      self.replay.load_state_dict(self.saved_replay)
      self.start_update = self.saved["update"]
      _restore_rng(self.saved["rng"])
    elif stage == "dagger" and args.init_mode != "scratch":
      source = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
      if source["stage"] != "bc" or source["teacher_hashes"] != self.teacher_hashes:
        raise ValueError("BC checkpoint stage or expert contents differ")
      self.student.load_state_dict(source["student"])
      if args.init_mode == "bc-with-demos":
        demo_path = args.bc_demo_checkpoint or args.bc_checkpoint
        demo_source = (torch.load(demo_path, map_location="cpu", weights_only=False)
                       if args.bc_demo_checkpoint else source)
        if demo_source["stage"] != "bc" or demo_source["teacher_hashes"] != self.teacher_hashes:
          raise ValueError("BC demo checkpoint stage or expert contents differ")
        self.replay.load_bc_demos(_replay_from_file(demo_path, demo_source))
        if any(min(counts) == 0 for counts in self.replay.bc_coverage().values()):
          raise ValueError("BC checkpoint lacks level coverage")
    self.writer = SummaryWriter(str(self.run_dir / "tensorboard")) if args.logger == "tensorboard" else None
    self.wandb = None
    if args.logger == "wandb":
      import wandb
      self.wandb = wandb.init(project=args.wandb_project, name=args.run_name,
                              dir=str(self.run_dir),
                              id=self.saved.get("wandb_run_id") if self.saved else None,
                              resume="allow" if self.saved else None,
                              config={"stage": stage, "num_envs": args.num_envs,
                                      "replay_capacity": args.replay_capacity})

  def close(self):
    for pool in getattr(self, "pools", {}).values():
      pool.close()
    if getattr(self, "writer", None):
      self.writer.close()
    if getattr(self, "wandb", None):
      self.wandb.finish()

  def collect(self):
    stats = {}
    with torch.inference_mode():
      for feature, pool in self.pools.items():
        observed, labeled, levels = [], [], []
        completions = torch.zeros((), device=self.device)
        falls = torch.zeros((), device=self.device)
        resets = torch.zeros((), device=self.device)
        for _ in range(self.args.rollout_steps):
          student_input = pool.obs["student"]
          teacher_action = pool.teacher(pool.obs)
          action = teacher_action if self.stage == "bc" else self.student(pool.obs)
          observed.append(student_input.detach().clone())
          labeled.append(teacher_action.detach().clone())
          if self.stage == "bc":
            levels.append(pool.terrain.terrain_levels.detach().clone())
          pool.obs, _, done, _ = pool.env.step(action)
          self.student.update_normalization(TensorDict({"student": student_input}, batch_size=[len(student_input)]))
          resets += done.sum()
          term = pool.env.unwrapped.termination_manager
          completions += term.get_term("rpl_feature_complete").sum()
          falls += term.get_term("fell_over").sum()
        observations = torch.cat(observed)
        actions = torch.cat(labeled)
        if self.stage == "bc":
          self.replay.add_bc(feature, observations, actions, torch.cat(levels))
        else:
          self.replay.add_dagger(feature, observations, actions)
        stats[f"{feature}/resets"] = int(resets.item())
        stats[f"{feature}/successes"] = int(completions.item())
        stats[f"{feature}/falls"] = int(falls.item())
        stats[f"{feature}/mean_level"] = float(pool.terrain.terrain_levels.float().mean())
        stats[f"{feature}/max_level"] = int(pool.terrain.terrain_levels.max().item())
        stats[f"{feature}/success_rate"] = (stats[f"{feature}/successes"] /
                                             max(stats[f"{feature}/resets"], 1))
        if self.stage == "bc":
          stats[f"{feature}/min_bc_samples_per_level"] = min(self.replay.bc_coverage()[feature])
    return stats

  def optimize(self, update):
    fraction = 1.0 if self.stage == "bc" else (
      bc_fraction(update - 1) if self.args.init_mode == "bc-with-demos" else 0.0)
    loss_sum = 0.0
    self.student.train()
    for _ in range(self.args.gradient_steps):
      observations, labels = self.replay.sample(self.args.batch_size, fraction, self.device)
      obs = TensorDict({"student": observations}, batch_size=[len(observations)])
      prediction = self.student(obs)
      loss = F.mse_loss(prediction, labels)
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
      self.optimizer.step()
      loss_sum += float(loss.detach())
    self.student.eval()
    return {"loss/action_mse": loss_sum / self.args.gradient_steps,
            "replay/bc_fraction": fraction}

  def save(self, update):
    folder = self.run_dir / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / f"student_{update}.pt"
    demo_path = folder / f"demos_{update}.pt"
    state = {"version": 2, "stage": self.stage, "update": update,
             "init_mode": self.args.init_mode if self.stage == "dagger" else None,
             "num_envs": self.args.num_envs, "teacher_paths": self.source_paths,
             "teacher_hashes": self.teacher_hashes,
             "learning_rate": self.args.learning_rate, "map_config": asdict(StudentMapCfg()),
             "student": self.student.state_dict(), "optimizer": self.optimizer.state_dict(),
             "demo_file": demo_path.name,
             "levels": {f: self.pools[f].terrain.terrain_levels.cpu().tolist() for f in FEATURES},
             "wandb_run_id": self.wandb.id if self.wandb else None,
             "rng": _rng_state()}
    demos = {"version": 2, "stage": self.stage, "update": update,
             "teacher_hashes": self.teacher_hashes, "replay": self.replay.state_dict()}
    _save_atomic(demo_path, demos)
    _save_atomic(path, state)
    print(f"Saved {path} and {demo_path}", flush=True)

  def train(self):
    if self.args.updates <= self.start_update:
      raise ValueError("--updates must exceed the checkpoint's completed updates")
    try:
      for update in range(self.start_update + 1, self.args.updates + 1):
        rollout_stats = self.collect()
        if self.stage == "bc" and any(min(c) == 0 for c in self.replay.bc_coverage().values()):
          raise RuntimeError("BC needs examples from all levels; increase --num-envs")
        metrics = {**rollout_stats, **self.optimize(update)}
        if self.writer:
          for key, value in metrics.items():
            self.writer.add_scalar(key, value, update)
        if self.wandb:
          self.wandb.log(metrics, step=update)
        print(f"{self.stage} {update}/{self.args.updates}: "
              f"MSE={metrics['loss/action_mse']:.5f}, "
              + ", ".join(f"{f} level={metrics[f'{f}/mean_level']:.2f} "
                          f"success={metrics[f'{f}/successes']}" for f in FEATURES), flush=True)
        if update % self.args.save_every == 0 or update == self.args.updates:
          self.save(update)
    finally:
      self.close()
