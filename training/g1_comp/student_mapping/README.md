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
algorithm. New cells are held as unobserved for 2 seconds, approximating the
slower buildup in the recorded ROS map. Each refresh also adds a smooth spatial
height bias (6.5 cm standard deviation at the coarse noise nodes), 1.5 cm
independent hit error, and 2 cm shared XY placement error. These are effective
training perturbations, not measured Livox sensor specifications. They affect
stored map hits, not the 693 policy values directly. All are configurable in
`StudentMapCfg`; setting the delay and noise standard deviations to zero
recovers the clean geometric prototype.

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
measure differences in the upstream ROS elevation mapper. The recorded ROS
comparison below checks that upstream difference on one fixed scene.

To replay a capture from `notebooks/collect_ros_data.ipynb` against the actual
student observation term on the same room-stairs scene:

```bash
cd training
python scripts/compare_student_ros_capture.py \
  ~/Downloads/20261125T220315Z_ros_mapping.tar.gz \
  --output-dir logs/ros_mapping_comparison
```

This requires the native mjlab Python environment but not ROS or Docker. It
reads the MCAP directly, reconstructs the pelvis from recorded LiDAR odometry
and joints, and raycasts `sim/g1_walk_scene.xml`. It writes `summary.json`,
per-step `steps.csv`, and `comparison.svg` with scan errors, observation-source
shares, and outbound/return stair-zone segments. The summary checks that the
reconstructed true scan and the ROS map projection match their recorded scans.
This script currently supports room-stairs captures with an uncompressed MCAP
and a policy driven by the true-terrain scan. ROS scan messages lack timestamps,
so it pairs them by bag record time and reports the pairing offsets. The student
still uses exact terrain ray hits; lower error than ROS does not prove it
reproduces LiDAR noise, fusion, or self filtering. Replay does not train or
change a policy.

On the 2026-11-25 room-stairs capture, the clean prototype had 3.8 cm mean
per-step scan RMSE against true terrain, versus 7.3 cm for ROS. With the
current defaults, student replay seeds 42–44 gave 7.5–7.9 cm. Mean disagreement
about which policy-scan cells were observed fell from 6.5% to 5.8–6.0%; in
the first quarter it fell from 17.5% to about 12.7%. This is a calibration on
**one** simulated scene and trajectory. The noise can still be too strong in
otherwise clear regions, and these numbers should not be treated as a
guarantee that ROS is more accurate at every point or on other terrain.

To measure mapping error while a saved expert walks a fixed level-9 lane:

```bash
python scripts/eval_student_map_rollout.py --task ascend \
  --checkpoint models/RPL/ascend/model_v5_1998_best.pt \
  --terrain-level 9 --steps 600
```

The expert receives the original exact scan for actions; the student scan is
measured alongside it. The report includes error by observation source and an
approximate feature-region breakdown (1.5–3.5 m past the spawn point).

Before the ROS-capture calibration, one 600-step, seed-42 rollout per level-9
task on 2026-11-25 gave:

| Task | Overall scan MAE | Feature-region MAE | Feature cells over 20 cm error |
| --- | ---: | ---: | ---: |
| Ascend | 8.4 cm | 3.0 cm | 6.7% |
| Descend | 6.5 cm | 5.4 cm | 9.1% |
| Ramps | 2.8 cm | 1.3 cm | 0.9% |
| Obstacles | 3.1 cm | 0.9 cm | 0.3% |

These are errors against exact MuJoCo terrain, not against a recorded ROS map.
In that clean-prototype run, the most reliable cells were those marked observed; inferred cells, especially
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
