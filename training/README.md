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

**2. Watch a trained robot.** Then open <http://localhost:8080>; `Ctrl+C` to quit:

```bash
python scripts/play.py G1-Comp-Course --viewer viser --checkpoint-file models/g1_comp_crouch/model_799.pt
```

**3. Train a new robot** (runs in the background):

```bash
setsid nohup python scripts/train.py G1-Comp-Rough --env.scene.num-envs 512 --agent.max-iterations 3000 --agent.logger tensorboard --agent.run-name walk1 > logs/walk1.log 2>&1 &
```

**4. See how far training got:**

```bash
grep -o "Learning iteration *[0-9]*/[0-9]*" logs/walk1.log | tail -1
```

**5. Stop training:**

```bash
pkill -f "[t]rain.py"
```

**6. Continue training from a saved model:**

```bash
setsid nohup python scripts/train.py G1-Comp-Rough --from logs/rsl_rl/g1_comp_rough/<folder>/model_<N>.pt --env.scene.num-envs 512 --agent.logger tensorboard --agent.run-name walk2 > logs/walk2.log 2>&1 &
```

First time on this computer? Run this once, in your `humanoid` folder, before step 1:

```bash
git pull && git switch competition && git submodule update --init
```

---

## Change the commands

| To... | Change this |
|---|---|
| Use a different world | `G1-Comp-Course` → `G1-Comp-Rough`, `G1-Comp-Crouch-Test` or `Unitree-G1-Flat` (see [Tasks](#tasks)) |
| Watch a different saved model | `--checkpoint-file <path to model_<N>.pt>` |
| Watch a world with no model | replace `--checkpoint-file ...` with `--agent zero` |
| Train more or fewer robots at once | `--env.scene.num-envs 512`: 256 if out of GPU memory, 4096 on an H100 |
| Train for longer | `--agent.max-iterations 3000` (about 4 s per iteration on a laptop) |
| Name the run | `--agent.run-name walk1`; also change `logs/walk1.log` to match |
| Start all robots on easy terrain | add `--env.scene.terrain.max-init-terrain-level 0` |
| Continue from any saved model | `--from <path to model_<N>.pt>`; the task can be different from the original one |
| Keep crouch and arm motion on when continuing (model past 2,000 iterations) | put `G1_COMP_SKILLS_START=0 G1_COMP_SKILLS_FULL=0` in front of `setsid` |
| Train in the foreground (watch the output, `Ctrl+C` to stop) | drop `setsid nohup` and everything from `>` onward |

**Viewer controls** (<http://localhost:8080>):
- **Walk:** Commands → Twist → Enable, then move the sliders.
- **Crouch:** Twist → Enable with all sliders at 0, then Base_height → Enable and move the height slider. The robot only crouches when standing still.
- **Respawn:** Reset Environment.

---

## Details

### Tasks

A task is the world the robot is put in: the terrain, the commands it gets, and the rewards while training. The same tasks are used for watching and for training.

| Task | World | Use it for |
|---|---|---|
| `G1-Comp-Rough` | Flat ground, slopes, rough ground, and stairs 5–23 cm high, getting harder as robots succeed. Walking first; crouching and arm motion phase in from iteration 500 to 2,000 | **The main one.** Trains the competition robot |
| `G1-Comp-Course` | The competition course, from the organizers' CAD | Practise and test on the real course, after Rough |
| `G1-Comp-Crouch-Test` | Flat, mostly standing still, crouching and arm motion from the start | Quick crouch check on a laptop. Never learns to walk |
| `Unitree-G1-Flat` | Unitree's original flat walking, with no height map | The old walking demo models only |

Models from any `G1-Comp-*` task work on every `G1-Comp-*` task. `Unitree-G1-Flat` models only work on `Unitree-G1-Flat`.

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
| `training/scripts/` | `play.py`, `train.py`, `eval_policy.py` (crouch and walking numbers), `check_task.py` (builds a task and steps it without a model, to check it works) |
| `training/g1_comp/` | Our tasks. Main settings are constants at the top of `env_cfgs.py` |
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
- **Skills ramp.** Standing still, crouching and arm motion phase in from iteration 500 to 2,000, after walking. Starting with them all on made the young robot learn to fall instead. When continuing a run, the ramp starts over (see the table above).
- **Starting exploration.** PPO's initial action noise is 0.5 instead of Unitree's 1.0, for the same reason.
- **One GPU per run.** Our `train.py` only works on one GPU; with several (`--gpu-ids`), the worker processes don't see our tasks.
