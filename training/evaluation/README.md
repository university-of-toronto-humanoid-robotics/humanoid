# Mobility route preview

This is a static, editable route proposal for the Mobility Trial on the
competition course. It belongs to the training/evaluation workflow. The script
reads `sim/competition/course_heightmap.npz` as source data and does not modify
the course, ROS stack, or simulation XML.

`mobility_route.json` lists CAD-frame waypoints in outbound order. The return
path follows the same points in reverse. The A–D target circles and ending
location are **estimates** from page 3 of the competition rules, not surveyed
CAD coordinates. `P1`–`P4` go around the open door leaf, `P6`–`P7` pass the
partition, and `P8`–`P10` line up with the stairs.

From the repository root, regenerate `mobility_route_preview.png` after edits:

```bash
.venv-mjlab/bin/python training/evaluation/preview_mobility_route.py
```

This needs NumPy, SciPy, and Matplotlib. The script prints waypoint coordinates
and samples each straight segment and the smoothed path against a 2D wall/fixture mask. Stairs,
landing, ramps, step-over beams, and the door sill are treated as walkable. The
current minimum sampled clearance is 0.28 m, above the provisional 0.20 m
screening margin. This centre-line check is not a full robot-body or 3D
collision guarantee. No policy or route controller runs in this stage.

## Run the policy on the route

From `training/`, with `.venv-mjlab` activated:

```bash
python scripts/eval_course_route.py --viewer viser
```

To watch without saving metrics or writing any evaluation files, add `--no-log`.

Open the local URL printed by the script. Viser uses the next available port
if 8080 is already occupied. The default checkpoint is
`models/best_g1_rough_heightmap/model_5996.pt`; use `--checkpoint PATH` to
choose another compatible model. The evaluator creates a dedicated
`G1-Course-HeightMap` task with the same actor observations and 29 joint actions
as `G1-Rough-HeightMap`. It loads the existing CAD course from `sim/competition/`
without modifying any ROS or course assets.
Use Viser's Reset Environment button for another trial, or press Ctrl+C in
the terminal to close the viewer and write the current trial summary.

The path uses short quadratic curves at clear via-point corners. Target-circle
centres, the door route, and the stair centreline remain fixed. The controller
uses the simulated robot pose and sends body-frame forward, lateral, and yaw
velocity commands at each 0.02 s policy step. At the ending location it turns
around and follows the same path back. The Viser route overlay and results
both use CAD-frame waypoint coordinates.

Results go to `training/logs/route_eval/<timestamp>/`: one CSV of step-by-step
pose, commands, progress, and path error, one JSON summary of time, falls, and
foot target hits, and a copy of the route JSON. A target hit requires a foot
site inside the estimated circle while that foot's terrain contact sensor
reports contact. The foot site is used as the location proxy for the sole.
The run stops on a fall, route completion, or the 300 s default simulated-time
limit. `--max-seconds` changes the limit; `--viewer none` runs without a browser.

For a quick CPU check:

```bash
python scripts/eval_course_route.py --viewer none --max-seconds 2
```
