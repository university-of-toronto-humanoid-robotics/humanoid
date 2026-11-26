# Competition course evaluation

This runs a compatible G1 height-map checkpoint through the competition course
along the predetermined outbound and return route in `mobility_route.json`. A
pure-pursuit-style look-ahead controller turns the route into velocity commands;
proportional feedback adjusts the forward command toward the target speed.

## Run with a checkpoint

From the repository root, activate the mjlab environment and move to `training/`:

```bash
source .venv-mjlab/bin/activate
cd training
```

For Viser, run the evaluator and open the local URL it prints:

```bash
python scripts/eval_course_route.py --viewer viser \
  --checkpoint models/best_g1_rough_heightmap/model_5996.pt
```

On macOS, use `mjpython` for a local MuJoCo window:

```bash
mjpython scripts/eval_course_route.py --viewer native \
  --checkpoint models/best_g1_rough_heightmap/model_5996.pt
```

Replace the checkpoint path with any checkpoint from the compatible
`G1-Flat-HeightMap`, `G1-Rough-HeightMap`, `G1-Open-HeightMap`, or
`G1-Open-Maze-HeightMap` family. The evaluator uses `G1-Course-HeightMap`, which
has the same policy observation and action layout.

| Argument | Meaning |
|---|---|
| `--checkpoint PATH` | Model to evaluate; defaults to `models/best_g1_rough_heightmap/model_5996.pt`. |
| `--viewer OPTION` | `viser` (browser, default), `native` (local MuJoCo window), or `none`. Use `mjpython` with `native` on macOS. |
| `--route PATH` | Route JSON; defaults to `evaluation/mobility_route.json`. |
| `--max-seconds N` | Simulated-time limit; defaults to 300 seconds. |
| `--output DIR` | Results directory; defaults to `logs/route_eval/<timestamp>/`. |
| `--no-log` | Visualize without saving evaluation files. |

With logging enabled, the evaluator saves step-by-step CSV data, a JSON trial
summary, and a copy of the route. The route waypoints can be edited in
`mobility_route.json`.

**Map input:** The policy currently receives a ground-truth terrain height map
from simulation. Observation noise and mapping errors are disabled in this
evaluation.

## Evaluate a distilled student

The student evaluator uses the same competition course, route, velocity
controller, and viewers. It loads a BC or DAgger `student_*.pt` checkpoint;
the separate `demos_*.pt` file is not needed for playback.

From `training/`, run either map mode in Viser:

```bash
python scripts/eval_student_course_route.py \
  --checkpoint /path/to/student_1000.pt --map-mode corrupted --viewer viser
python scripts/eval_student_course_route.py \
  --checkpoint /path/to/student_1000.pt --map-mode clean --viewer viser
```

On macOS, replace `python` with `mjpython` and use `--viewer native` for a
local MuJoCo window. Add `--no-log` to visualize without writing results.
The script also accepts `--viewer none`, `--route`, `--max-seconds`, `--seed`,
and `--output`, as in the expert evaluator. Logged results go under
`training/logs/student_route_eval/` by default.

`corrupted` uses the same 3 Hz, partially observed, delayed, noisy elevation
mapping and noisy joint observations as student training. `clean` turns off
observation corruption and supplies exact terrain scans in both of the CNN's
map channels. It is a diagnostic upper bound: the second channel is another
exact current scan, not a delayed historical measurement. Both modes retain
the student's 1,600-input observation layout and 29 joint actions.
