"""Save pictures of a task's world (no robots, no GPU simulation): one from
straight above and one from an angle. From training/ inside the container:

  python scripts/view_world.py Stairs                 # -> logs/worlds/Stairs_top.png, Stairs_side.png
  python scripts/view_world.py Open --train           # the full training grid instead of the play sample

To walk around a world instead, play it with no model:
  python scripts/play.py Stairs --agent zero --viewer viser
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import _setup  # noqa: E402,F401
import mujoco  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.terrains.terrain_entity import TerrainEntity  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task")
  parser.add_argument("--train", action="store_true", help="the training grid (default: the play sample)")
  parser.add_argument("--close", type=float, nargs=2, metavar=("X", "Y"),
                      help="a close-up looking at this point instead (<task>_close.png)")
  parser.add_argument("--out", default="logs/worlds")
  parser.add_argument("--width", type=int, default=1600)
  parser.add_argument("--height", type=int, default=1000)
  args = parser.parse_args()

  terrain_cfg = load_env_cfg(args.task, play=not args.train).scene.terrain
  terrain_cfg.num_envs = 1
  spec = TerrainEntity(terrain_cfg, device="cpu").spec
  spec.visual.global_.offwidth, spec.visual.global_.offheight = args.width, args.height
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  gen = terrain_cfg.terrain_generator
  span = max(gen.num_rows * gen.size[0], gen.num_cols * gen.size[1]) if gen is not None else 10.0
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (0.0, 0.0, 0.0)
  import imageio.v2 as imageio

  views = [("top", -90.0, 90.0, 1.25 * span, (0.0, 0.0, 0.0)), ("side", -35.0, 60.0, 1.1 * span, (0.0, 0.0, 0.0))]
  if args.close is not None:  # one tile, from an angle
    views = [("close", -30.0, 35.0, 9.0, (args.close[0], args.close[1], 0.3))]
  for name, elevation, azimuth, distance, lookat in views:
    camera.elevation, camera.azimuth, camera.distance = elevation, azimuth, distance
    camera.lookat[:] = lookat
    renderer.update_scene(data, camera)
    path = out / f"{args.task}_{name}.png"
    imageio.imwrite(path, renderer.render())
    print(f"saved {path}  ({model.ngeom} geoms)")


if __name__ == "__main__":
  main()
