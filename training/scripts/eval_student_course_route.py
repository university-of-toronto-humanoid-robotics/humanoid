"""Run a distilled student checkpoint on the competition route.

From training/:
  python scripts/eval_student_course_route.py --checkpoint path/to/student_1000.pt --map-mode corrupted
  mjpython scripts/eval_student_course_route.py --checkpoint path/to/student_1000.pt --map-mode clean --viewer native
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _setup  # noqa: F401  (registers the course task)
import torch

from evaluation.route import ROUTE, CourseMap, build_paths, load_route, start_before_waypoint
from evaluation.student_course import student_course_env_cfg
from evaluation.student_map_viewer import StudentMapOverlay, StudentMapPolicy
from g1_comp.student_actor import TemporalStudentActor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg

from eval_course_route import (NativeRouteViewer, RouteEvaluation, RouteViewer, TASK,
                               TRAINING, configure_course_spawn)


class StudentRouteViewer(StudentMapOverlay, RouteViewer):
  """Overlay the student's actual input map on the competition route."""

  map_color = (255, 80, 220)  # distinguish it from the cyan outbound route


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--checkpoint", type=Path, required=True,
                      help="BC or DAgger student_*.pt weights checkpoint")
  parser.add_argument("--map-mode", choices=("clean", "corrupted"), default="corrupted")
  parser.add_argument("--route", type=Path, default=ROUTE)
  parser.add_argument("--start-waypoint", help="Start before an outbound route waypoint, e.g. P8")
  parser.add_argument("--start-lead-m", type=float, default=0.5,
                      help="Distance before the chosen waypoint; 0 starts at it (default: 0.5)")
  parser.add_argument("--viewer", choices=("viser", "native", "none"), default="viser")
  parser.add_argument("--max-seconds", type=float, default=300.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--no-log", action="store_true")
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Student checkpoint not found: {args.checkpoint}")
  if args.max_seconds <= 0:
    parser.error("--max-seconds must be positive")

  route = load_route(args.route)
  course = CourseMap.load()
  outbound, returning, smoothed = build_paths(route, course)
  start = (start_before_waypoint(route, course, outbound, args.start_waypoint, args.start_lead_m)
           if args.start_waypoint else None)
  print(f"Student map: {args.map_mode}; route: {outbound.length:.2f} m out, "
        f"{returning.length:.2f} m back; smoothed corners: {', '.join(smoothed) or 'none'}")
  torch.manual_seed(args.seed)
  cfg = student_course_env_cfg(args.map_mode)
  if start is not None:
    configure_course_spawn(cfg, course, start)
    print(f"Starting {args.start_lead_m:.2f} m before {args.start_waypoint} at "
          f"CAD ({start.xy[0]:.2f}, {start.xy[1]:.2f}), heading {start.yaw:.2f} rad")
  if args.viewer == "viser":
    # The normal terrain dots are ground truth, not the student's input map.
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.debug_vis = False
  cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"),
                           clip_actions=agent_cfg.clip_actions)
  try:
    obs = env.get_observations()
    policy = TemporalStudentActor(obs, {"student": ["student"]}, "student").to("cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("stage") not in ("bc", "dagger") or "student" not in saved:
      raise ValueError("Expected a BC or DAgger student_*.pt checkpoint")
    policy.load_state_dict(saved["student"], strict=True)
    policy.eval()
    output = (args.output or TRAINING / "logs" / "student_route_eval" /
              (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.map_mode))
    evaluator = RouteEvaluation(env, StudentMapPolicy(policy), route, course, outbound, returning,
                                output, args.max_seconds, record_metrics=not args.no_log,
                                start_s=start.progress_s if start else 0.0)
    evaluator.checkpoint = args.checkpoint.resolve()
    evaluator.route_path = args.route.resolve()
    try:
      if args.viewer == "viser":
        viewer = StudentRouteViewer(evaluator)
        print(f"Open http://localhost:{viewer._server.get_port()} to see the route and robot.")
        viewer.run()
      elif args.viewer == "native":
        NativeRouteViewer(evaluator).run()
      else:
        with torch.inference_mode():
          while evaluator.status == "running":
            evaluator.step()
    finally:
      evaluator.finish("stopped")
  finally:
    env.close()


if __name__ == "__main__":
  main()
