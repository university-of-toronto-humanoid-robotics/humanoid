# Student elevation map prototype

This is an opt-in observation for the future four-expert student. It replaces
only the actor's 693-value terrain scan in a copy of an RPL task configuration.
The actor stays at 791 values, the critic at 806, and the action space at 29.
Expert training tasks and checkpoints are unchanged. This package does not
train or distill a student yet.

The map is a world-aligned 5 × 5 m, 5 cm grid, matching
`ros2_ws/src/bringup/config/g1_elevation_mapping.yaml`. A 10 cm terrain probe
provides candidate hits across that area. The observation keeps hits outside
the 0.6 m self-filter radius, within the Mid-360's approximate 52° downward
field of view and 5 m range, and not hidden by a sampled terrain surface. It
remembers seen heights as the robot moves and refreshes at 3 Hz. Each 50 Hz
policy step samples the 33 × 21 pelvis-yaw-aligned scan from the current map.
Unknown cells follow the ROS `height_scan/scan.py` order: near the feet use the
estimated sole height; farther out use inpainted map height when available;
otherwise use the foot estimate. The scan is pelvis height above ground,
clipped to [−1, 2] m.
Inpainting runs on the full 5 × 5 m map at each map refresh, before the
policy window is sampled. This placement matters at the scan boundary.

This is a **geometric approximation**, not the actual ROS mapper. The probe
uses exact simulated terrain before visibility filtering. It does not reproduce
individual Livox rays, FAST-LIO pose drift, the robot's full self-filter mesh,
CuPy elevation variance and Bayesian update, or the ROS filter's exact inpaint
algorithm. Its height readings are currently clean. We should compare its
observed/foot/inpaint source shares and height errors against recorded ROS maps
before using it as the sole student training input.

From `training/`, run a one-environment CPU check:

```bash
python scripts/check_student_map.py --task ascend --steps 25
```

Use `--task descend`, `ramps`, or `obstacles` for the other experts. The JSON
output gives actor/critic shape validation, finite-value validation, mean
absolute error against the exact terrain scan, source shares, mean map age,
and throughput. It applies zero actions, so it is a mapping smoke check rather
than a locomotion evaluation. On a CUDA host, benchmark candidate scale with
`--device cuda:0 --num-envs 1024`, then `4096`; report peak allocated GPU
memory and environment steps per second. Start with a small run, since the
extra 51 × 51 terrain probe and 100 × 100 memory map add cost per environment.
The unit tests compare the fallback calculation with the ROS `height_scan`
implementation on a synthetic map and compare complete simulated scans to the
same ROS calculation before and after a map refresh:

```bash
python tests/test_student_mapping.py
```

To print the complete scan comparison for a terrain task:

```bash
python scripts/compare_student_ros_scan.py --task descend --steps 25
```

This uses the **same simulated map** for both calculations. It verifies map
sampling, inpaint placement, foot fallback, and scan ordering; it does not
measure differences in the upstream ROS elevation mapper. A true end-to-end
comparison still needs a recorded ROS GridMap and synchronized robot poses on
the same terrain.

To measure mapping error while a saved expert walks a fixed level-9 lane:

```bash
python scripts/eval_student_map_rollout.py --task ascend \
  --checkpoint models/RPL/ascend/model_v5_1998_best.pt \
  --terrain-level 9 --steps 600
```

The expert receives the original exact scan for actions; the student scan is
measured alongside it. The report includes error by observation source and an
approximate feature-region breakdown (1.5–3.5 m past the spawn point).

On 2026-11-25, one 600-step, seed-42 rollout per level-9 task gave:

| Task | Overall scan MAE | Feature-region MAE | Feature cells over 20 cm error |
| --- | ---: | ---: | ---: |
| Ascend | 8.4 cm | 3.0 cm | 6.7% |
| Descend | 6.5 cm | 5.4 cm | 9.1% |
| Ramps | 2.8 cm | 1.3 cm | 0.9% |
| Obstacles | 3.1 cm | 0.9 cm | 0.3% |

These are errors against exact MuJoCo terrain, not against a recorded ROS map.
The most reliable cells were those marked observed; inferred cells, especially
inpainted cells near stairs, had larger errors. The ROS CuPy mapper also uses
OpenCV Telea inpainting and measurement fusion, while this prototype uses
local averaging and exact terrain candidates. The current results do **not**
validate the prototype as a faithful ROS mapping replacement for stair-policy
distillation. Repeating ascent and descent for seeds 43 and 44 gave similar
overall errors: 8.3–8.4 cm for ascent and 6.2–6.5 cm for descent across seeds
42–44. In the feature region, 6.0–6.7% of ascent cells and 8.4–9.1% of descent
cells exceeded 20 cm error. Compare recorded ROS layers on the same terrain
before relying on this prototype's error distribution.

Use `attach_student_map(rpl_env_cfg("ascend"))` to build a student-view config
for a future DAgger trainer. The expert should continue to read the original
RPL observation from the same simulator state; the student view is separate.
No policy architecture, curriculum, or expert checkpoint needs changing for
that comparison.
