# ros2 workflow

How the G1's software fits together, and how to test all of it without the robot.

## The rule: the sim replaces the hardware, and nothing else

The real G1 gives us three things and takes one:

| Topic | Type | From |
|---|---|---|
| `/lowstate` | `unitree_hg/LowState` | The robot: joint angles and speeds, and the pelvis IMU |
| `/livox/lidar`, `/livox/imu` | Livox `CustomMsg`, `Imu` | The Mid-360 on its head |
| `/lowcmd` | `unitree_hg/LowCmd` | **To** the robot: a target angle, gains and torque for each motor |

`mujoco_g1_walk_sim` publishes and obeys exactly these, so every node above it is the one that runs on the robot:

```
 hardware (sim:=true MuJoCo, sim:=false the G1)        software, the same either way (bringup/g1.launch.py)

  /lowstate ──────────────▶ joint_state_bridge ──▶ /joint_states ──▶ robot_state_publisher ──▶ TF
  /livox/lidar, /livox/imu ▶ gravity_align → FAST-LIO → self_filter → elevation mapping ──▶ height_scan_node
                                                                                                  │ /height_scan
  /lowstate ──────────────▶ walk_controller ◀────────────────────────────────────────────────────┘
  /lowcmd   ◀────────────── walk_controller ◀── /cmd_vel (where to walk: the sim's motion script, later teleop or a planner)
```

- **`walk_controller`** (`policy/`) runs the trained policy 50 times a second. It does nothing until `/walk_controller/start` is called; then it eases the joints to the standing pose, and hands over to the policy (a height-map policy first waits for `/height_scan`). `/walk_controller/stop`, a fall, or losing `/lowstate` makes it go limp. `/walk_controller/state` says which of `wait`, `stand`, `walk`, `damp` it is in.
- **The sim also plays the gantry and the operator,** since the real robot needs both. The robot starts hanging from a gantry (the policy's gains can't balance it standing still). The sim calls `start` once FAST-LIO is running, lets go of the gantry when the controller reports `walk`, waits `hold` seconds, then publishes the scenario's motion script on `/cmd_vel`.
- **For testing only,** the sim also publishes `/clock`, `/ground_truth/odom` and `/ground_truth/height_scan`, and ends with a report: FAST-LIO's error against the true pose, and `/height_scan`'s against the true terrain.

There is a second, simpler sim for perception alone:

| Node | Robot | Use it to |
|---|---|---|
| `mujoco_g1_walk_sim` | Physics. Stands in for the robot's hardware, set up by a scenario file | Test the whole stack: perception under real walking motion, and how a policy copes with a scene |
| `mujoco_mid360_sim` | Posed along a scripted circle with a gait-like bob, no physics, no controller | Test perception on its own, with motion controlled exactly. Run it with `perception.launch.py` |

## Setup

- **Build** with `--symlink-install`, from `ros2_ws/`: `colcon build --symlink-install`. The sims find scenes, scenarios and checkpoints by walking up from their source files, which only works with symlinks.
- **Python packages** for the sims:
  - `mujoco==3.14.0`, `mujoco-lidar==0.3.5`, `warp-lang==1.12.0`, `scipy`, `pyyaml`, and `torch` (only to read the policy checkpoint, so the CPU build is enough).
  - The Docker image has them all. Only for users of native Ubuntu 24.04, install `mujoco-lidar` and `warp-lang` with `pip install --no-deps --target .sim-deps ...` and add that folder to `PYTHONPATH`, so pip doesn't replace ROS's numpy.
- **Elevation mapping needs an NVIDIA GPU** (CuPy). The Docker image has its packages on x86_64; run it with the GPU passed through (`--gpus all`). On a Mac it can't run. FAST-LIO and the sims run anywhere; without a GPU, warp traces the LiDAR on the CPU, more slowly.

## Running a test

1. **Launch it.** One command starts the sim, perception (with RViz) and the controller:
   ```bash
   ros2 launch bringup g1.launch.py                                   # scenario room_walk
   ros2 launch bringup g1.launch.py scenario:=course_ramps            # the competition course's ramps
   # a terrain-aware policy, fed the live elevation map as on the robot:
   ros2 launch bringup g1.launch.py scenario:=room_stairs_heightmap scan:=elevation_map
   ```
   - `scan:=true_terrain` feeds a height-map policy the true terrain instead (`/ground_truth/height_scan`), to tell a policy's failures from perception's. Without `scan`, the scenario's own `policy.scan` is used.
   - `policy_type:=` and `checkpoint:=` override the scenario's policy. `viewer:=false` and `rviz:=false` run it without windows.
   - Everything stops when the scenario ends.
2. **What happens.** The robot hangs on the gantry while FAST-LIO initialises and `gravity_align` levels `odom` from the first second of IMU data. About a second later the sim starts the controller, which stands the robot up (2 s), waits for a terrain scan if its policy needs one, and walks. The gantry lets go, and after `hold` seconds (3 s) the motion script begins.
   - To run the perception-only sim instead: `ros2 launch bringup perception.launch.py use_sim_time:=true`, then `ros2 run bridge mujoco_mid360_sim` in a second terminal.
3. **Watch it in RViz** (fixed frame `odom`): the G1 model, FAST-LIO's path and registered cloud, and the elevation map (`/elevation_mapping_node/elevation_map_filter`).
   - Look over each filter, such as `inpaint`, `elevation_map`, etc..
4. **Read the result.** At the end the sim prints a line like:
   ```
   FAST-LIO vs ground truth over 51.7 m, 1226 poses: translation RMSE 0.029 m, max 0.042 m, final 0.030 m | rotation RMSE 0.26 deg, max 0.54 deg, final 0.14 deg
   ```
   Both are compared in FAST-LIO's own `camera_init` frame, so they measure odometry drift, independent of how `odom` is levelled. The walk sim also logs where `odom` sits in the scene at startup, which is what you need to compare the elevation map with a known height map.
5. **With the perception-only sim, restart perception before the next run.** Sim time starts from zero again, and FAST-LIO and the elevation map keep their old map otherwise.

## Scenarios (`mujoco_g1_walk_sim`)

A scenario is a YAML file in `sim/scenarios/`, chosen with `scenario:=<name>` or `scenario:=<path>.yaml`. Paths in it are relative to the repo root.

```yaml
scene: sim/g1_course_scene.xml                     # a MuJoCo scene that includes description/g1/g1_29dof_mjlab.xml
spawn: {x: 9.0, y: 3.5, yaw: 3.141592653589793}    # pelvis position (m) and heading (rad); height comes from the ground
duration: 150.0                                    # s of sim time, at most
policy:
  type: rsl_rl_g1_flat                             # policy/policy/walk_policy.py: walk_controller runs it
  checkpoint: training/models/g1_walk/model_12400.pt
motion:
  type: waypoints                                  # bridge/motion.py
  speed: 0.4                                       # m/s
  points: [[7.9, 3.6], [6.8, 3.6], [6.8, 4.7]]     # (x, y) in the scene frame
```

| Scenario | Scene | What it does |
|---|---|---|
| `room_walk` (default) | `sim/g1_walk_scene.xml`: a walled room with boxes, pillars and four 20 cm steps | A 52 m loop past the stairs and boxes, on flat ground |
| `course_ramps` | `sim/g1_course_scene.xml`: the competition course (`sim/competition/course.xml`) | Starts in the step-over section facing the 15° crossing ramps, then two laps of a 1.1 m square on them |
| `course_ramps_heightmap` | The competition course | A height-map policy crossing the ramps from the deck side, round a rectangle and back |
| `room_stairs_heightmap` | The LiDAR room | A height-map policy up the room's 20 cm steps to the third, then back down |

To make a new test, copy a scenario and change it:
- **Scene:** a new scene must include `description/g1/g1_29dof_mjlab.xml` (see `sim/g1_course_scene.xml`), and its own geoms must stay in geom group 0. The spawn height and the fall check only look at that group.
- **Motion types:**
  - `waypoints` (`points`, `speed`, `laps`): turns toward each point in turn and walks when roughly facing it.
  - `velocity`: open-loop `segments`, each `{duration, vx, vy, wz}`.
- **Policy types,** both read from an rsl_rl checkpoint:
  - `rsl_rl_g1_flat`: mjlab's `Unitree-G1-Flat` actor, 98 observations, blind.
  - `rsl_rl_g1_heightmap`: the team's `G1-Flat-HeightMap` and `G1-Rough-HeightMap` actors, 791 observations: the same 98 plus a 693-point terrain scan. The scenario's `policy.scan` (or `scan:=`) picks its source: `true_terrain` ray-casts the scene as training does (`bridge/height_scan.py`), and `elevation_map` is `/height_scan` from `height_scan_node`, which samples the live elevation map at the pelvis pose from TF, exactly as on the robot. The `G1-Comp-*` tasks (793 observations) are a different family and aren't supported yet.
- **New types** are a class each, registered in the `*_TYPES` dict at the bottom of `bridge/bridge/motion.py` or `policy/policy/walk_policy.py`.
- **Navigation uses the true pose,** not FAST-LIO's, so perception errors don't change where the robot goes.
- **The policies' trained command ranges** are vx −0.5 to 1.0 m/s, vy ±0.5 m/s and wz ±0.5 rad/s. `walk_controller` clips `/cmd_vel` to them.

## Parameters

`g1.launch.py` takes `sim`, `scenario`, `scan`, `policy_type`, `checkpoint`, `hold`, `viewer` and `rviz` (`ros2 launch bringup g1.launch.py --show-args`). The sims' own parameters, for `ros2 run ... --ros-args -p name:=value`:

| Parameter | Default | Sim | Meaning |
|---|---|---|---|
| `scenario` | `room_walk` | walk | Scenario name or path |
| `scene`, `duration`, `speed` | from the scenario | walk | Override the scenario's scene, run length and waypoint speed |
| `viewer` | `true` | both | Open the MuJoCo viewer |
| `duration` | 60 s | scripted | Run length |
| `hold` | 3 s | both | Walk: the wait between the controller starting to walk and the motion script. Scripted: standing still at the start, for FAST-LIO and `gravity_align` |
| `rays_per_slice` | 2000 | both | Rays per 10 ms slice; 2000 gives the Mid-360's 200,000 points/s |
| `seed` | 0 | both | Seeds IMU and LiDAR noise |
| `noise` | `true` | both | LiDAR noise on or off; `false` gives ideal scans |
| `range_noise` | 0.02 m | both | 1σ Gaussian noise along each ray |
| `angle_noise` | 0.15° | both | 1σ direction error per axis |
| `dropout` | 0.02 | both | Fraction of returns lost |
| `blind_zone` | 0.1 m | both | Closer returns are dropped, with noise on or off |

The noise values come from the Mid-360's spec sheet. IMU noise comes from the noise settings on the MuJoCo sensors in `description/g1/g1_29dof_rev_1_0.xml`.

## Reference results

Measured with seed 0 and the configs in this repo (FAST-LIO 0.2 m voxels), on an RTX 3070. Use them to spot regressions.

| Run | FAST-LIO translation RMSE / max | Rotation RMSE | Elevation map |
|---|---|---|---|
| `mujoco_mid360_sim`, `noise:=false` | 0.6 / 0.9 cm | 0.08° | |
| `mujoco_mid360_sim` | 2.2 / 2.8 cm | 0.11° | |
| `room_walk`, `noise:=false` | 2.2 / 3.1 cm | 0.24° | Step heights within 0.1 cm |
| `room_walk` | 2.0 / 2.8 cm | 0.22° | 13 of 13 waypoints |
| `course_ramps` | 3.0 / 4.5 cm | 0.93° | Ramp tiles 2.1 cm RMSE against the CAD height map, apart from cells next to tall objects |
| `room_stairs_heightmap` | 1.0 / 2.3 cm | 0.27° | Policy on the true scan: up and down, 2 of 2 waypoints |
| `room_stairs_heightmap`, `scan:=elevation_map` | 1.2 / 2.5 cm | 0.27° | Policy on the live map: up and down; its scan 7.4 cm RMSE from the true one |
| `course_ramps_heightmap` | 1.3 / 3.4 cm | 0.35° | Policy on the true scan: round the ramps, 5 of 5 waypoints |
| `course_ramps_heightmap`, `scan:=elevation_map` | 1.4 / 5.3 cm | 0.52° | Policy on the live map: round the ramps; its scan 11.3 cm RMSE from the true one |

The `room_walk`, `room_stairs_heightmap` and `course_ramps_heightmap` rows were measured through `g1.launch.py` (the policy in `walk_controller`, one run each, RTX 4060 Ti). The `noise:=false` and `course_ramps` rows are from before that change, when the policy ran inside the sim.

What the runs show:
- **The flat policy can't climb the 15° ramps at 0.4 m/s:** it walks in place at their foot. At `speed:=1.0` it gets up, but runs away downhill, overshoots and can't turn on the slope. It didn't fall in either case.
- **The elevation map smears tall objects** (walls, the counter) 5–10 cm into the neighbouring cells, with errors up to about 1 m there.
- **A height-map policy can walk on the live map,** but only with two fixes, and each run above is a single run:
  - `self_filter` removes the robot's own legs and arms from the scan. Without it they enter the map as a trail of obstacles at pelvis height, and the robot fell within 2 s of walking.
  - The upside-down Mid-360 can't see the ground within about 0.94 m of the robot, which is most of the policy's scan. The map remembers what it saw from farther away; for cells it never saw, `height_scan_node` (`perception/height_scan/height_scan/scan.py`) uses the feet's height under the robot and the map's inpainted layer beyond that. Inpainting everywhere filled the hole near a wall with the wall's height, and the robot fell standing; using the feet everywhere read a drop beside a step as flat, and it walked off the side.
- **The policy was trained on near-perfect scans** (±2 cm noise), while the live scan's worst point per step is typically 20–80 cm off (edges, smeared walls). Training with map-like errors, or a sensor that sees the ground near the feet, would make this much more robust.
- **`G1-Rough-HeightMap` trained on stairs, slopes and bumps, not walls:** at `course_ramps`' spawn, beside the lane wall, door frame and beams, it falls while standing, even on the true scan.

## Where things are

| File | What |
|---|---|
| `bridge/bridge/mujoco_mid360_sim.py` | Scripted sim. Also has the Mid-360 tracing, noise model, IMU and ground truth that the walk sim reuses |
| `bridge/bridge/mujoco_g1_walk_sim.py` | Walk sim: the G1's hardware (`/lowcmd` in, `/lowstate` and the LiDAR out), plus the gantry and the scenario's operator |
| `bridge/bridge/g1_model.py` | Builds the G1 the way mjlab does for training (actuators, armature, contacts, solver), places it at the spawn, and applies motor commands |
| `bridge/bridge/scenario.py`, `bridge/bridge/motion.py` | Scenario loading and motion types |
| `bridge/bridge/height_scan.py` | The sim's true terrain scan, and its check of `/height_scan` against it |
| `bridge/bridge/joint_state_bridge.py` | `/lowstate` to `/joint_states`, on the robot and in the sim |
| `policy/policy/walk_controller.py` | The controller node: states, safety checks, `/lowcmd` |
| `policy/policy/walk_policy.py`, `policy/policy/g1.py`, `policy/policy/lowcmd.py` | Policy types; the G1's joint order, standing pose and gains; the checksum the robot wants on `/lowcmd` |
| `interfaces/unitree_hg/` | Unitree's `LowState` and `LowCmd` messages |
| `perception/height_scan/` | `height_scan_node`: the elevation map as the policies' terrain scan, on `/height_scan`, for the robot and the sim |
| `bridge/bridge/gravity_align.py`, `bridge/bridge/body_to_pelvis.py`, `bridge/bridge/self_filter.py` | Started by `perception.launch.py`: the levelled `odom -> camera_init`, the link from FAST-LIO's TF tree to the URDF's, and the filter that removes the robot's own body from the scan before elevation mapping |
| `../../sim/scenarios/`, `../../sim/*.xml` | Scenarios and scenes |
| `bringup/launch/g1.launch.py` | Everything: the hardware (sim or real), then the bridge, perception and the controller |
| `bringup/launch/perception.launch.py` | The perception stack by itself |
| `perception/fast_lio/config/mid360.yaml`, `bringup/config/g1_elevation_mapping.yaml` | FAST-LIO and elevation mapping settings |
