# Train and run the G1

## Quick start (copy, paste)

**1. Turn on the container.** Docker Desktop must be running. In your `humanoid` folder:

```bash
docker compose -f docker/compose.windows.yaml up -d --build
docker compose -f docker/compose.windows.yaml exec dev bash
```

Then, inside the container:

```bash
source /opt/mjlab/bin/activate && cd training
```

**2. Look at a world** (no model: the robot just stands, then falls). Then open <http://localhost:8080>; `Ctrl+C` to quit. Any task name from [Tasks](#tasks) works:

```bash
python scripts/play.py Stairs --agent zero --viewer viser
```

**3. Train the base model: walking on flat ground** (runs in the background; about 7 s per iteration on a laptop, so 3,000 iterations take about 6 h):

```bash
setsid nohup python scripts/train.py Flat --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --agent.run-name base > logs/flat_base.log 2>&1 &
```

**4. See how far training got:**

```bash
grep -o "Learning iteration *[0-9]*/[0-9]*" logs/flat_base.log | tail -1
```

**5. Stop training:**

```bash
pkill -f "[t]rain.py"
```

**6. Fine-tune the base model on another task** (`Open` = everything on the course; or `Ramp`, `Stairs`, ...). The run name and log file are yours to choose:

```bash
setsid nohup python scripts/train.py Open --from logs/rsl_rl/flat/<folder>/model_<N>.pt --env.scene.num-envs 1024 --agent.max-iterations 5000 --agent.logger tensorboard --agent.run-name open1 > logs/open1.log 2>&1 &
```

**7. Watch a trained model** on any task:

```bash
python scripts/play.py Open --viewer viser --checkpoint-file logs/rsl_rl/open/<folder>/model_<N>.pt
```

The same `--from` line continues any run after it was stopped (use the newest `model_<N>.pt`, and lower `--agent.max-iterations` to the number left). On a laptop (8 GB GPU), don't go above 4096 robots on `Flat` or 1024 on the other tasks: past that the GPU runs out of memory and training gets about 10× slower instead of crashing.

First time on this computer? Run this once, in your `humanoid` folder, before step 1:

```bash
git pull && git switch competition && git submodule update --init
```

---

## Change the commands

| To... | Change this |
|---|---|
| Use a different world | the task name: `Flat`, `Ramp`, `Stairs`, `Open`, `Rough`, `VeryRough`, `Competition`, or any with `-Crouch` added (see [Tasks](#tasks)) |
| Watch a different saved model | `--checkpoint-file <path to model_<N>.pt>` |
| Watch a world with no model | replace `--checkpoint-file ...` with `--agent zero` |
| See what the robot sees | add `--fpv` (see [First-person view](#first-person-view-and-ros)) |
| Train more or fewer robots at once | `--env.scene.num-envs`: laptop max 4096 on `Flat`, 1024 on the others |
| Train for longer | `--agent.max-iterations 5000` |
| Name the run | `--agent.run-name base`; also change `logs/flat_base.log` to match |
| Let robots start on harder rows | add `--env.scene.terrain.max-init-terrain-level 5` (default 1: they earn the harder rows) |
| Continue from any saved model | `--from <path to model_<N>.pt>`; the task can be different from the original one |
| Train without mirror augmentation, to compare | put `G1_MIRROR=0` in front of `setsid` |
| Train with Unitree's full contacts, to compare | put `G1_FULL_CONTACT=1` in front of `setsid` |
| Train in the foreground (watch the output, `Ctrl+C` to stop) | drop `setsid nohup` and everything from `>` onward |
| Save pictures of a world | `python scripts/view_world.py Stairs` → `logs/worlds/` |
| Save a video of a model | `python scripts/record_video.py Open --checkpoint <model_<N>.pt>` → `logs/videos/` |

**Viewer controls** (<http://localhost:8080>):
- **Walk:** Commands → Twist → Enable, then move the sliders.
- **Crouch** (models trained on a `-Crouch` task): Twist → Enable with all sliders at 0, then Base_height → Enable and move the height slider. The robot only crouches when standing still.
- **Respawn:** Reset Environment.

### First-person view and ROS

In the competition the operator may not look at the robot, only at what it sees. To practise that, and to test perception later, the simulated G1 has a camera where its RealSense D435i is (in the head, pitched 47.6° down, as in Unitree's URDF).

**In the viewer.** Add `--fpv` to any play command; the image (color and depth) is under *Camera Feeds* in the viewer's side panel:

```bash
python scripts/play.py Open --fpv --viewer viser --checkpoint-file logs/rsl_rl/open/<folder>/model_<N>.pt
```

**On ROS 2.** Add `--ros` instead. The robot then also follows `/cmd_vel`, like the real one will:

```bash
python scripts/play.py Open --ros --viewer viser --num-envs 1 --checkpoint-file logs/rsl_rl/open/<folder>/model_<N>.pt
```

In two more terminals (each: `docker compose -f docker/compose.windows.yaml exec dev bash`):

```bash
ros2 run rqt_image_view rqt_image_view /g1/head_camera/image_raw
```

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

| Topic | Type | What |
|---|---|---|
| `/g1/head_camera/image_raw` | `sensor_msgs/Image`, `rgb8`, 640×360, up to 30 Hz | The head camera |
| `/g1/head_camera/depth/image_raw` | `sensor_msgs/Image`, `32FC1` | Its depth, in meters |
| `/g1/head_camera/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics (42° vertical field of view) |
| `/cmd_vel` | `geometry_msgs/Twist` | Forward and sideways speed (m/s) and turning rate (rad/s). The robot stops 0.5 s after the last message |

The real camera looks steeply down, so you see the floor from about 0.5 to 2.5 m ahead and never the horizon. To try another angle: `G1_FPV_PITCH=20` in front of the command (degrees below level). Works on the new tasks only.

---

## Details

### Tasks

A task is the world the robot is put in: the terrain, the commands it gets, and the rewards while training. The same tasks are used for watching and for training.

**The tasks to use** (defined in `g1_comp/tasks.py`, worlds in `g1_comp/worlds.py`). A model from any of them runs, and continues training, on any other.

| Task | World | Use it for |
|---|---|---|
| `Flat` | Flat ground. The robot still gets its height map (it reads flat) | **The base model.** Learns to walk, turn, stop and follow commands. Train this first; everything else starts from it with `--from` |
| `Ramp` | Walled lanes filled with the course's ramp tiles (0.61 m squares): as a "hill" (up, down, up, down) or each tile facing its own way. From 4° on the easiest row to 17° on the hardest (course: 15°) | Fine-tune on ramps only, to see what ramps need |
| `Stairs` | Narrow walled lanes (0.77 to 1.5 m) with stairs up to a landing and down again. From 2 steps of 6 cm on 34 cm treads to 5 steps of 21.5 cm on 24 cm treads (course: 5 × 20.3 cm on 24 cm) | Fine-tune on stairs only, up and down |
| `Open` | Endless lanes with every part of the course in random order: ramps, stairs, step-over beams (straight, diagonal, V), the door sill and door posts. Lanes 0.77 to 2.44 m wide. A quarter of the lanes are all stairs, a quarter all ramps | **The generalist.** One model for the whole course; fine-tune the base here |
| `Rough` | Open fields of realistic rough ground: bumps up to 4 cm, rolling ground, slopes up to 14°, curbs up to 8 cm | Robustness to floors that aren't perfectly flat |
| `VeryRough` | Open fields much rougher than anything expected: 12 cm bumps, 27° slopes, 20 cm waves and steps, scattered blocks | Stress test; optional extra robustness |
| `Competition` | The competition course itself, from the organizers' CAD. Robots start anywhere along it | The last fine-tune, and the place to test |
| `<Task>-Crouch` (e.g. `Open-Crouch`) | The same world as `<Task>`. A quarter of the robots stand still (instead of a tenth), and 60% of those are told to crouch, down to a 0.42 m pelvis height | Add crouching to a model that already walks there |

On every row of a lane world the parts get harder, and robots move up a row when they cover 3.5 m, down when they cover less than half of what they were told. In lanes most robots are told to walk along the lane (either way); on `Flat`, `Rough` and `VeryRough` in any direction.

What each robot is doing at any moment while training: walking a commanded velocity (90%), standing still (10%). Arms are moved by random motion instead of the policy for half of the standing time and 5% of the walking time (as the IK will do for button presses).

**Older tasks** (kept in `g1_comp/env_cfgs.py`; their models don't run on the tasks above, and the other way round):

| Task | World | Status |
|---|---|---|
| `G1-Comp-Flat` | Flat ground, walking only | Replaced by `Flat`. Never trained beyond a speed test |
| `G1-Comp-Rough` | mjlab's rough mix (slopes, bumps, waves) with wide pyramid stairs up to 23 cm. Standing (30%), crouching and arm takeover (30%) phase in on a fixed schedule, iterations 500 to 2,000 | Replaced by `Open`. Runs `rough1` and `rough2` never learned to walk here: the extra skills arrived before the robots could walk |
| `G1-Comp-Course` | The competition course, with the `G1-Comp-Rough` robot setup | Replaced by `Competition` |
| `G1-Comp-Crouch-Test` | Flat, 60% of robots standing, crouching and arm takeover from the start | Quick crouch check. `models/g1_comp_crouch/model_799.pt` came from it: crouches, doesn't walk |
| `Unitree-G1-Flat`, `Unitree-G1-Rough` | Unitree's original tasks, no changes (Flat has no height map) | `models/g1_walk/` came from `Unitree-G1-Flat` |

### The training plan

1. **Base: `Flat`, about 3,000 iterations.** Walks after about 500; by 3,000 the curves are flat (10,000 would add nothing). One base model for the whole team.
2. **Generalist: `Open --from <base>`.** One model that does ramps, stairs, beams and the door. This is the competition model: the robot can't switch models mid-course.
3. **Polish: `Competition --from <generalist>`**, a few hundred to a thousand iterations, then test there.
4. **Crouching, if needed: `Open-Crouch --from <generalist>`**, then `Competition-Crouch`.

`Ramp` and `Stairs` are for working on one part at a time: fine-tune the base there to find out what that part needs (rewards, speeds, how far up the rows robots get), then bring what you learn into `Open`. A model fine-tuned only on `Stairs` forgets ramps, so the specialists are for learning, not for the competition.

How to tell a stage is done: on `Flat`, *Mean episode length* at 1000 and *track_linear_velocity* above 0.7. On the lane worlds, *Curriculum/terrain_levels* stops climbing; rows 8-9 are a little harder than the course.

### Train on Google Colab

Open [`colab/train_g1.ipynb`](colab/train_g1.ipynb) in Colab (File → Open notebook → GitHub, or upload it), pick a GPU runtime, and run the cells top to bottom. It installs the same versions as the Docker image, keeps the runs on your Google Drive, continues after a disconnect, and records a video of the result. It runs the code that is pushed to GitHub, so push first. The notebook says how many robots fit on each Colab GPU.

### What the new tasks do differently from the G1-Comp-* ones

- **No skills ramp.** The old tasks switched on standing (30%), crouching and arm takeover (30%) between iterations 500 and 2,000 whether or not the robots could walk yet. In both long runs they couldn't, and episode length fell from there (`rough2`: 277 steps at iteration 2,000, 118 at 5,000). Now standing is 10% from the start, crouching lives in the `-Crouch` tasks, and the arm takeover happens mostly while standing.
- **The arm takeover no longer drifts.** While the arms are overridden the policy's own arm outputs do nothing, so nothing held them in place and they wandered (`rough2`'s symmetry number went from 0.002 to 0.76 as the takeover came in). A small penalty (`arm_idle`) now keeps them near the default pose.
- **Height map relative to the feet** (`scan.py`). Each value is the ground height at that point minus the height of the soles: 0 on flat ground, 0.2 for a 20 cm step. The old one was "pelvis height above the point". **On the robot:** map height at the point − (pelvis height in the map − pelvis height above the soles from the leg joints); points with no data read 0.
- **Fixed input scale for the map, the crouch command and the arm flag** (`policy.py`). RSL-RL rescales inputs by their running spread, which is near zero for the map on flat ground and exactly zero for a crouch command that never changes. A `Flat` model would then see its first stairs as numbers 5-20× larger than anything before. These inputs now have the same scale in every task, and a new network starts with zero weight on the map.
- **Lane commands and curriculum** (`lanes.py`). In lane worlds 90% of robots are told to walk along the lane (heading off by up to ±14°, plus sideways speed, as an operator would steer). Robots move up a row after covering 3.5 m from where they started.
- **Speeds stay at the course's.** Forward up to 1.0 m/s on open ground and 0.8 m/s in lanes; Unitree's task widens to 2 m/s after 5,000 iterations.
- **30% of robots get a fixed turning rate** (a joystick) instead of a heading to face. Unitree: all get a heading.
- **Fewer contact checks** (see `tasks.py`, `LEGS_ONLY_WORLD_CONTACT`). Only the legs and feet can touch the world; the upper body still collides with the robot itself.

### Saved models

| Where | What | In git? |
|---|---|---|
| `training/models/` | Models we keep and share. To share one, copy it here. | yes |
| `training/logs/rsl_rl/<experiment>/<date>_<run name>/` | Everything a training run writes | no |

What's in `training/models/` now:

| File | Task | Does |
|---|---|---|
| `models/g1_comp_crouch/model_799.pt` | `G1-Comp-*` | Stands and crouches on command (down to ~0.56 m), even with the arms moving. **Doesn't walk.** |
| `models/g1_walk/model_<N>.pt` (N = 500 … 12400) | `Unitree-G1-Flat` | Unitree's flat walking demo. The early ones show how learning progresses |

What a training run's folder holds:
- `<experiment>` is `g1_comp_rough`, `g1_comp_course`, `g1_comp_crouch_test` or `g1_velocity`.
- `model_<N>.pt`: a saved model every 100 iterations, plus the last one. **This is what you watch or continue from.**
- `policy.onnx`: the latest model, exported for the robot.
- `params/`: every setting the run used.
- `events.out.tfevents...`: the learning curves.

### Is training working?

Learning curves: run this, then open <http://localhost:6006>:

```bash
tensorboard --logdir logs/rsl_rl --bind_all
```

- *Mean episode length* going up: the robot falls less. If it drops to about 15 steps and stays there, it learned to fall on purpose; stop the run and investigate.
- *track_linear_velocity* going up: it's learning to walk.
- *Curriculum/terrain_levels* going up (0–9): it's handling harder terrain.
- *Curriculum/assist/weight_share* (only with the assist force on): share of the robot's weight the assist force carries. Should fall to 0 as terrain levels pass 4; if it stays at the top, the robots are leaning on it.
- *Episode_Reward/foot_on_edge* and *foot_stumble*: small negative numbers; closer to 0 means safer steps.
- *Mean symmetry loss* in the log: how differently the policy acts on mirrored situations (only reported, not trained on). Should shrink.
- To test a model while training continues, watch the newest `model_<N>.pt` with `--num-envs 4`.
- For numbers instead of eyeballing (crouch accuracy, walking speed):

  ```bash
  python scripts/eval_policy.py --checkpoint <model_<N>.pt>
  ```
- Training stops if the laptop sleeps or shuts down, or Docker Desktop quits. You lose at most the last 99 iterations.

### Long runs on the cluster

Thousands of iterations with 4096 robots go on the H100 cluster with Slurm. See [`cluster/README.md`](cluster/README.md).

### Where things are

| Path | What |
|---|---|
| `training/scripts/` | `play.py`, `train.py`, `eval_policy.py` (crouch and walking numbers), `check_task.py` (builds a task and steps it without a model, to check it works), `check_features.py` (checks the sim-to-real features below are wired right; run after changing them), `view_world.py` (pictures of a world), `record_video.py` (video of a model, no viewer needed) |
| `training/g1_comp/` | Our tasks. New ones: `tasks.py` (main settings are the constants at its top), `worlds.py` (the 3D worlds), `lanes.py`, `scan.py`, `policy.py`, `camera.py`. Older `G1-Comp-*` ones: `env_cfgs.py`, `terrains.py`. Shared: `sensors.py`, `mdp.py`, `commands.py`, `actions.py`, `map_noise.py`, `symmetry.py` |
| `training/colab/` | The Google Colab notebook |
| `training/models/` | Saved models we share |
| `training/logs/` | Training runs (not in git) |
| `training/cluster/` | Slurm scripts |
| `training/unitree_rl_mjlab/` | Unitree's code (a git submodule, never edited) |
| `sim/competition/` | The competition course for MuJoCo, its 1 cm height map, and the measured dimensions ([README](../sim/competition/README.md)) |
| `tools/cad/` | Rebuilds the course from the organizers' CAD |

To see the course alone, run this from the `humanoid` folder (`--spawn stairs` adds a standing G1):

```bash
python3 sim/competition/view_course.py --no-robot
```

### What our tasks change, and why

All built on Unitree's `Unitree-G1-Rough`. Unchanged: the robot, the motor gains, most rewards, randomization and PPO.

- **Height map (`sensors.py`).**
  - A 1.6 × 1.0 m grid of 33 × 21 points (5 cm spacing) around the robot, from 0.5 m behind to 1.1 m ahead, turning with its heading.
  - Each value is the pelvis height above that point in meters, clipped to [−1, 2]. −1 means too tall to climb.
  - The rays start 2 m above the pelvis. Unitree's start at pelvis height, so stairs and walls taller than the pelvis looked like holes.
  - **On the robot, perception must provide exactly this.**
- **Foot clearance (`mdp.py`).** Measured above the ground under each foot. Unitree's measured from world height zero, which punished being up the stairs.
- **Crouch (`commands.py`).**
  - A pelvis-height command from 0.785 m down to 0.42 m. It's only used while standing still and changes by at most 0.25 m/s.
  - The matching crouch pose is added to the leg targets, so the robot only has to balance and correct. **The robot-side code must add the same offsets** (`crouch_leg_pitch`).
- **Arm motion (`actions.py`).**
  - About 30% of the time, the arms follow random motion instead of the robot's own commands. This teaches the legs to keep balance while the IK uses the arms.
  - A 0/1 input tells the robot when this is happening. **On the robot, set it to 1 while the IK drives the arms.**
- **Sim-to-real features.** Settings are constants at the top of `env_cfgs.py`; papers and pages in the table below.
  - **Map noise (`map_noise.py`).** The policy sees the scan the way a SLAM map gets it wrong (the critic sees it clean; off in `play.py`):
    - *per-episode shift:* the whole map off by up to ±2 cm sideways/forward and ±3 cm up/down, fixed for one robot's life (drift, calibration);
    - *per point:* every cell, every step, up to ±3 cm (LiDAR noise);
    - *missing:* 2% of cells each step;
    - *delay:* 0–60 ms old;
    - *blink:* the whole map gone for 0.1–0.3 s, about every 20 s.
    - **On the robot, fill missing cells and dropouts with "flat ground at the feet": the pelvis height above the soles.**
  - **Assist force (off by default).** An upward push on the pelvis carrying up to 50% of the robot's weight, fading to 0 as the robots' average terrain level reaches 4. Training on `G1-Comp-Rough` only. Turn it on with `G1_COMP_ASSIST=0.5`.
    - *Why it's off:* in a 512-robot laptop run (`rough1`, 3,000 iterations), no robot ever got past level 0. So the force stayed at 50% the whole run, and the robots fell once `play.py` took it away.
    - *Untested:* neither paper compares a run without it for this kind of force. A2CF's force pushes mostly sideways and is randomly switched off 20% of the time (p.3).
    - *To find out:* compare a plain run with an `ASSIST=0.5` run on the cluster (see `cluster/README.md`).
  - **Mirror augmentation (`symmetry.py`).** PPO also trains on every sample mirrored left↔right. If you add an observation, add its mirror rule there (it stops with an error otherwise).
  - **Command delay and motor gains.** Joint commands arrive 0–30 ms late, and each joint's PD gains are scaled 0.9–1.1; both drawn per episode. Training only: `play.py` and `eval_policy.py` show the nominal robot.
  - **Steadier torso.** Torso tilt-rate penalty doubled (−0.05 → −0.1): the LiDAR is in the head.
  - **Unsafe stepping.** Small penalties per foot per step: −0.2 for standing on a step edge (more than 5 cm between neighbouring points under the sole), −0.3 for a foot pushed sideways by the ground (a toe hitting a riser). Our 24 cm treads leave 3.5 cm around the foot. Small on purpose: climbing still pays more than the penalty costs. If *foot_on_edge* stays large and robots stop climbing stairs, lower `EDGE_WEIGHT`.
  - **Continuing a model trained before these features** (e.g. `models/g1_comp_crouch/model_799.pt`) works. The critic's contact-force input changed frame (world → robot heading), so it re-adapts over the first iterations.
- **Skills ramp.** Standing still, crouching and arm motion phase in from iteration 500 to 2,000, after walking. Starting with them all on made the young robot learn to fall instead. When continuing a run, the ramp starts over (see the table above).
- **Starting exploration.** PPO's initial action noise is 0.5 instead of Unitree's 1.0, for the same reason.
- **One GPU per run.** Our `train.py` only works on one GPU; with several (`--gpu-ids`), the worker processes don't see our tasks.

### Why each sim-to-real feature (papers, PDF page numbers)

| Feature | Paper | Result | Page |
|---|---|---|---|
| Map noise, delay | Gallant, arXiv 2511.14625 | Real G1: without LiDAR noise/delay in training, success "drops significantly" on ground terrain | p.4 (model), p.8 (Fig. 6), p.13 (failures) |
| | VB-Com, arXiv 2502.14814v2 | At 100% map noise: 80.5% of goals when trained with noise vs 48.7% without (sim) | p.8, Table II |
| | Miki et al. 2022, arXiv 2201.08117 | Per-point / per-episode noise scopes; three noise conditions | p.12 |
| Assist force (off by default) | A2CF, arXiv 2506.23125 | G1 29-DoF, stairs to 23 cm: top terrain level at ~4k iterations vs ~10k. Their force is mostly sideways, randomly switched off 20% of the time | p.6 (Fig. 4 on p.5); p.3, p.5 |
| | Watch Your Step, arXiv 2610.29612 | The schedule we use (50% → 0 by level 4), with 8,192 robots; no comparison without it | p.4 |
| Mirror augmentation | Mittal et al. 2024, arXiv 2403.04359 | Higher return (15.5 → 17.5), far more symmetric; fewer perception failures on hardware | p.6, Table I |
| Command delay | arXiv 2610.32277 (mjlab, sim only) | 40 ms delay: 95% survive vs 15% without delay training | p.1, p.4 |
| Motor gains | Harbin HPC 2503.00692; Gallant | Standard ±20% (we use ±10%) | p.5 Table I; p.13 |
| Steadier torso | Gallant; UniPoint, arXiv 2609.23666 | Torso tilt and vibration degrade the LiDAR map | p.8; p.7 |
| Map blink | CAP, arXiv 2609.11553 | Small but positive: 97.9% vs 97.5% under noise; recovers faster when blind | p.18, Table A8, Fig. A1 |
| Unsafe stepping | arXiv 2603.07928 (G1 + Mid-360) | Near-100% safe steps on stairs in sim; learned faster than without | HTML version, Sec. IV |
