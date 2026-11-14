"""Evaluate a rough height-map checkpoint on the competition Mobility Trial route.

From training/:
  python scripts/eval_course_route.py --viewer viser
  python scripts/eval_course_route.py --viewer none --max-seconds 2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from datetime import datetime
import json
from pathlib import Path
import traceback

import _setup  # noqa: F401  (registers the local tasks)
import numpy as np
import torch

from evaluation.route import (
  COURSE_START_XY,
  ROUTE,
  CourseMap,
  RouteFollower,
  build_paths,
  load_route,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel


TRAINING = Path(__file__).resolve().parents[1]
TASK = "G1-Course-HeightMap"
DEFAULT_CHECKPOINT = TRAINING / "models" / "best_g1_rough_heightmap" / "model_5996.pt"


class RouteEvaluation:
  """Owns one trial, including velocity commands, foot targets, and logs."""

  def __init__(self, env, policy, route: dict, course: CourseMap,
               outbound, returning, output: Path, max_seconds: float,
               record_metrics: bool = True) -> None:
    self.env = env
    self.base = env.unwrapped
    self.policy = policy
    self.route = route
    self.course = course
    self.outbound = outbound
    self.returning = returning
    self.output = output
    self.max_seconds = max_seconds
    self.record_metrics = record_metrics
    self.robot = self.base.scene["robot"]
    self.foot_contacts = self.base.scene["feet_ground_contact"]
    self.twist = self.base.command_manager.get_term("twist")
    self.foot_ids, _ = self.robot.find_sites(("left_foot", "right_foot"))
    self.origin_xy = self.base.scene.env_origins[0, :2].cpu().numpy().copy()
    self.trial_no = 0
    self._csv_file = None
    self.start_trial()

  def cad_xy(self) -> np.ndarray:
    world_xy = self.robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
    return world_xy - self.origin_xy + COURSE_START_XY

  def _write_command(self, vx: float, vy: float, yaw_rate: float) -> None:
    self.twist.time_left.fill_(1e9)
    self.twist.is_heading_env.fill_(False)
    self.twist.is_standing_env.fill_(False)
    self.twist.vel_command_b[0] = torch.tensor(
      [vx, vy, yaw_rate], device=self.base.device, dtype=self.twist.vel_command_b.dtype
    )

  def start_trial(self) -> None:
    self.trial_no += 1
    self.follower = RouteFollower(self.route, self.outbound, self.returning)
    self.status = "running"
    self.step_count = 0
    self.error_sum = 0.0
    self.error_square_sum = 0.0
    self.error_max = 0.0
    self.last_cross_track = 0.0
    self.progress = 0.0
    self.next_goal = "A1"
    self.foot_hits: dict[str, list[dict]] = {"outbound": [], "return": []}
    self._seen: dict[str, set[str]] = {"outbound": set(), "return": set()}
    if self.record_metrics:
      self.output.mkdir(parents=True, exist_ok=True)
      (self.output / "route.json").write_text(json.dumps(self.route, indent=2) + "\n")
      self._csv_file = (self.output / f"trial_{self.trial_no:03d}.csv").open("w", newline="")
      self.writer = csv.writer(self._csv_file)
      self.writer.writerow(("time_s", "phase", "cad_x", "cad_y", "heading_rad", "progress",
                            "cross_track_m", "next_goal", "cmd_vx", "cmd_vy", "cmd_yaw"))
    self._write_command(0, 0, 0)

  def _check_foot_targets(self) -> None:
    phase = self.follower.phase
    if phase not in self.foot_hits:
      return
    foot_xy_w = self.robot.data.site_pos_w[0, self.foot_ids, :2].detach().cpu().numpy()
    foot_xy = foot_xy_w - self.origin_xy + COURSE_START_XY
    contact_found = self.foot_contacts.data.found
    assert contact_found is not None
    touching_ground = contact_found[0].detach().cpu().numpy() > 0
    points = self.route["waypoints"]
    targets = points[1:] if phase == "outbound" else list(reversed(points[:-1]))
    for p in targets:
      if p["kind"] not in {"goal", "start", "end"} or p["id"] in self._seen[phase]:
        continue
      within = np.linalg.norm(foot_xy - [p["x"], p["y"]], axis=1) <= self.route["goal_radius_m"]
      touched = np.flatnonzero(touching_ground & within)
      if len(touched):
        self._seen[phase].add(p["id"])
        event = {"goal": p["id"], "foot": ("left", "right")[int(touched[0])],
                 "time_s": round(self.step_count * self.base.step_dt, 3)}
        self.foot_hits[phase].append(event)
        print(f"[{phase}] foot contacted {p['id']} at {event['time_s']:.2f} s")

  def step(self) -> None:
    if self.status != "running":
      return
    cad_xy = self.cad_xy()
    if self.record_metrics:
      self._check_foot_targets()
    yaw = float(self.robot.data.heading_w[0].item())
    command = self.follower.update(cad_xy, yaw)
    self.progress = command.progress
    self.next_goal = command.next_goal
    if command.phase == "complete":
      self.finish("complete")
      return
    self._write_command(command.vx, command.vy, command.yaw_rate)
    with torch.no_grad():
      obs = self.env.get_observations()  # after writing twist, so policy sees this command
      action = self.policy(obs)
      _, _, dones, _ = self.env.step(action)
    self.step_count += 1
    error = command.cross_track
    self.last_cross_track = error
    if self.record_metrics:
      self.error_sum += error
      self.error_square_sum += error * error
      self.error_max = max(self.error_max, error)
      self.writer.writerow((round((self.step_count - 1) * self.base.step_dt, 3), command.phase,
                            round(float(cad_xy[0]), 4), round(float(cad_xy[1]), 4),
                            round(yaw, 4), round(command.progress, 5), round(error, 4),
                            command.next_goal, round(command.vx, 4), round(command.vy, 4),
                            round(command.yaw_rate, 4)))
    if bool(dones[0].item()):
      self.finish("fall" if bool(self.base.reset_terminated[0].item()) else "env_timeout")
    else:
      if self.record_metrics:
        self._check_foot_targets()
      if self.step_count * self.base.step_dt >= self.max_seconds:
        self.finish("timeout")

  def finish(self, status: str) -> None:
    if self.status != "running":
      return
    self.status = status
    self._write_command(0, 0, 0)
    if not self.record_metrics:
      print(f"Trial {self.trial_no}: {status}")
      return
    n = max(1, self.step_count)
    summary = {
      "status": status,
      "checkpoint": str(self.checkpoint),
      "route": str(self.route_path),
      "time_s": round(self.step_count * self.base.step_dt, 3),
      "falls": int(status == "fall"),
      "progress_fraction": round(self.progress, 4),
      "mean_cross_track_m": round(self.error_sum / n, 4),
      "rms_cross_track_m": round((self.error_square_sum / n) ** 0.5, 4),
      "max_cross_track_m": round(self.error_max, 4),
      "foot_hits": self.foot_hits,
    }
    (self.output / f"trial_{self.trial_no:03d}_summary.json").write_text(
      json.dumps(summary, indent=2) + "\n"
    )
    if self._csv_file is not None:
      self._csv_file.close()
      self._csv_file = None
    print(f"Trial {self.trial_no}: {status}, progress {self.progress:.1%}, "
          f"RMS path error {summary['rms_cross_track_m']:.2f} m")
    print(f"Results: {self.output}")

  def reset_trial(self) -> None:
    self.finish("manual_reset")
    self.start_trial()


class RouteViewer(ViserPlayViewer):
  def __init__(self, evaluator: RouteEvaluation) -> None:
    super().__init__(evaluator.env, evaluator.policy, verbosity=VerbosityLevel.INFO)
    self.evaluator = evaluator

  def setup(self) -> None:
    super().setup()
    ev = self.evaluator
    origin = ev.origin_xy
    for name, path, color in (("outbound", ev.outbound, (0, 220, 255)),
                              ("return", ev.returning, (255, 110, 0))):
      xy = path.xy[::10]
      if not np.array_equal(xy[-1], path.xy[-1]):
        xy = np.vstack((xy, path.xy[-1]))
      z = ev.course.heights_at(xy) + 0.07
      xyz = np.column_stack((xy - COURSE_START_XY + origin, z))
      self._server.scene.add_line_segments(
        f"/fixed_bodies/mobility_route/{name}",
        np.stack((xyz[:-1], xyz[1:]), axis=1), colors=color,
        thickness=0.025 if name == "outbound" else 0.014,
      )
    for p in ev.route["waypoints"]:
      if p["kind"] not in {"goal", "start", "end"}:
        continue
      centre = np.array([p["x"], p["y"]])
      angle = np.linspace(0, 2 * np.pi, 41)
      xy = centre + ev.route["goal_radius_m"] * np.column_stack((np.cos(angle), np.sin(angle)))
      z = ev.course.heights_at(xy) + 0.09
      xyz = np.column_stack((xy - COURSE_START_XY + origin, z))
      self._server.scene.add_line_segments(
        f"/fixed_bodies/mobility_route/goal_{p['id']}",
        np.stack((xyz[:-1], xyz[1:]), axis=1), colors=(255, 225, 55), thickness=0.014,
      )
      self._server.scene.add_label(
        f"/fixed_bodies/mobility_route/label_{p['id']}", p["id"],
        position=(float(centre[0] - COURSE_START_XY[0] + origin[0]),
                  float(centre[1] - COURSE_START_XY[1] + origin[1]),
                  float(ev.course.heights_at(centre) + 0.22)),
      )
    with self._server.gui.add_folder("Mobility route"):
      self._route_info = self._server.gui.add_html("Starting route evaluation...")
    self._refresh_info()

  def _refresh_info(self) -> None:
    ev = self.evaluator
    info = (
      f"<b>Status:</b> {ev.status}<br>"
      f"<b>Phase:</b> {ev.follower.phase}<br>"
      f"<b>Next target:</b> {ev.next_goal or '—'}<br>"
      f"<b>Progress:</b> {ev.progress:.1%}<br>"
    )
    if ev.record_metrics:
      info += (
        f"<b>Path error:</b> {ev.last_cross_track:.2f} m<br>"
        f"<b>Foot target contacts:</b> {sum(map(len, ev.foot_hits.values()))}<br>"
      )
    info += f"<b>Time:</b> {ev.step_count * ev.base.step_dt:.1f} s"
    self._route_info.content = info

  def _execute_step(self) -> bool:
    try:
      self.evaluator.step()
      self._step_count += 1
      self._stats_steps += 1
      if self.evaluator.step_count % 25 == 0 or self.evaluator.status != "running":
        self._refresh_info()
      if self.evaluator.status != "running":
        self.pause()
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(self._last_error, VerbosityLevel.SILENT)
      self.pause()
      return False

  def reset_environment(self) -> None:
    super().reset_environment()
    self.evaluator.reset_trial()
    self._refresh_info()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--route", type=Path, default=ROUTE)
  parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
  parser.add_argument("--max-seconds", type=float, default=300.0)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--no-log", action="store_true", help="Visualize without recording metrics or writing files")
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.max_seconds <= 0:
    parser.error("--max-seconds must be positive")

  route = load_route(args.route)
  course = CourseMap.load()
  outbound, returning, smoothed = build_paths(route, course)
  print(f"Route: {outbound.length:.2f} m out, {returning.length:.2f} m back; "
        f"smoothed corners: {', '.join(smoothed) or 'none'}")
  cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"),
                           clip_actions=agent_cfg.clip_actions)
  try:
    runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    output = args.output or TRAINING / "logs" / "route_eval" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluator = RouteEvaluation(env, policy, route, course, outbound, returning,
                                output, args.max_seconds, record_metrics=not args.no_log)
    evaluator.checkpoint = args.checkpoint.resolve()
    evaluator.route_path = args.route.resolve()
    try:
      if args.viewer == "viser":
        viewer = RouteViewer(evaluator)
        print(f"Open http://localhost:{viewer._server.get_port()} to see the route and robot.")
        viewer.run()
      else:
        while evaluator.status == "running":
          evaluator.step()
    finally:
      evaluator.finish("stopped")
  finally:
    env.close()


if __name__ == "__main__":
  main()
