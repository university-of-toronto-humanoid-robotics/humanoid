# G1 fall recovery (standing up): HoST in our MuJoCo stack

The walking policies can't get up after a fall. This folder trains the policies that
can, following **HoST** ("Learning Humanoid Standing-up Control across Diverse
Postures", Huang et al., RSS 2025, [arXiv 2502.08378](https://arxiv.org/abs/2502.08378),
[code](https://github.com/OpenRobotLab/HoST)), which stood a real G1 up from the
ground, platforms, walls and slopes. Everything for it lives here; the walking tasks
(`g1_comp/`) and the shared scripts are untouched.

**Two policies, picked by the IMU.** The paper trains one policy per terrain and found
that training "on your back" and "face down" together made both worse. So, like the
paper's prone experiment, we keep two families: **supine** (back, sides, sitting
against a wall) and **prone** (face down, sides). After a fall the robot uses the
pelvis IMU: body-frame gravity x < 0 means it is on its back -> supine policy, else
prone. Side-lying starts are rolled 50-90 deg from their family's posture, so each
policy learns the sides that end up on its side of that switch. `HoST-All` (one
policy for everything) is there as an experiment.

## Tasks

All have the same inputs (456 = 6 frames x 76) and outputs (23 joints), so a model
of one continues on any other (`--from`).

| Task | Robots start | Paper? |
|---|---|---|
| `HoST-Ground` | on the back, flat ground | yes |
| `HoST-Prone` | face down, flat ground | yes (their `ground_prone`) |
| `HoST-Platform` | on the back on a 22-46 cm platform, legs over the edge | yes |
| `HoST-Wall` | sitting, pushed back against a 14-84 deg incline | yes |
| `HoST-Slope` | on the back on a 1-15 deg slope, head uphill | yes |
| `HoST-Stairs` | on the back or a side across the course's stairs (20 cm risers), any direction | ours |
| `HoST-Supine-Mixed` | the supine family on flat (back + sides), platform, wall, ramps to 17 deg (head up or down), stairs | ours: **the supine policy to use** |
| `HoST-Prone-Mixed` | face down + sides on flat, ramps, stairs | ours: **the prone policy to use** |
| `HoST-All` | everything, plus random falls, in one policy | experiment |

## Commands (from `training/` in the container)

```bash
python g1_host/scripts/train.py HoST-Ground --env.scene.num-envs 1024          # train (laptop: smoke tests only)
python g1_host/scripts/play.py HoST-Ground --viewer viser --checkpoint-file logs/rsl_rl/host_ground/<run>/model_<N>.pt
python g1_host/scripts/play.py HoST-Supine-Mixed --agent zero --viewer viser    # see the worlds, robots just fall
python g1_host/scripts/eval.py --task HoST-Ground --checkpoint <model> [--robustness] [--video]
python g1_host/scripts/eval.py --gallery --supine <model> --prone <model> --video
python g1_host/scripts/check.py all        # wiring checks, no model needed
python g1_host/scripts/check_ppo.py        # PPO math vs HoST's code, no simulator needed
python g1_host/scripts/bench.py HoST-Ground --num-envs 1024 2048               # robots per GPU
```

Then open <http://localhost:8080> for the viewer. Play and eval use the final action
bound (0.25) and no pull force, as the paper's evaluation.

**Real training happens on the H100 cluster:** see [`cluster/README.md`](cluster/README.md)
(one page, written for whoever submits the jobs). Laptop speed (RTX 4060, `bench.py`):
13,000 steps/s at 1,024 robots (4 s per iteration), 16,500 at 2,048 using 2 GB of GPU
memory, so 4,096-16,384 robots fit an 80 GB H100 with lots of room; a 12,000-iteration
policy takes most of a day on the laptop.

`eval.py --gallery` is the "does it work everywhere" test: 19 scenarios (flat on the
back / face down / each side / random fall, platform, walls at 30 and 60 deg, slopes
and ramps head up / head down / sideways, stairs head up / head down / sideways / face
down / random fall), 250 robots each with randomization, success rate and time per
scenario, and an mp4 per scenario with `--video` (`logs/videos/host/`).

## What to look at while training (TensorBoard: `tensorboard --logdir logs/rsl_rl`)

- `Curriculum/force_and_bound/pull_force`: the upward help, 200 N at the start. It drops
  20 N each time a batch of robots ends an episode with the head above 0.9 m. Should reach 0.
- `Curriculum/force_and_bound/beta`: the action bound, 1.0 -> 0.25 on the same rule
  (limits how fast the joints can move: smooth, not violent, get-ups).
- `Curriculum/force_and_bound/success_rate`: robots standing at the end of their episode.
- `Reward_group/task|regu|style|post`: one critic per group.
- `Loss/l2c2_policy`, `Loss/l2c2_value`: the smoothness regularization.

## How the paper maps onto our code

| HoST | Here |
|---|---|
| Observation: ang vel, gravity, q, qd, last action, beta; 6-frame history | `mdp.host_proprio` (history 6, oldest first) |
| Action: target = q + beta a, PD 200 Hz, policy 50 Hz | `actions.HostJointAction` |
| 4 critics (task, regu, style, post), weights 2.5 / 0.1 / 1 / 1 | `algorithm.py`, `runner.group_rewards` |
| Task reward = product of head height and orientation | `runner.group_rewards` (task group) |
| L2C2 smoothness (actor + critics) | `algorithm.HostPPO.update` |
| Pulling force 200 N + action bound curriculum | `actions._apply_wrench`, `events.force_and_bound` |
| Every reward of Tab. VI, per-terrain weights | `mdp.py`, `tasks._rewards` |
| Domain randomization of Tab. II | `tasks._events`, `events.randomize_actuators` |
| Terrains (ground, platform, wall, slope) | `terrains.py` |
| Actor 512-256-128 + tanh, critics 512-256, PPO settings | `params.py`, `algorithm.py` |
| Evaluation protocol and metrics | `scripts/eval.py` |

Every number is in [`params.py`](params.py) with the line of HoST's code or the paper
table it comes from.

### Where we follow HoST's code instead of the paper's text

The released code and the paper disagree in a few places; the code is what produced
the paper's results (their README thanks someone for finding "typos in the paper"), so
we use it. The paper's value is next to each constant in `params.py`.

| | Paper | Code (used) |
|---|---|---|
| Joint velocity penalty | -1e-4 | -1e-3 |
| Joint tracking error penalty | -0.25 | -2.5e-4 |
| L2C2 weights (actor, critics) | 1, 0.1 | 0.111, 0.0111 (computed from its bounds) |
| CoM offset, vertical | +-8 cm | +-6 cm |
| Control delay | 0-100 ms | 0-20 ms |
| Platform heights | 20-92 cm | 22-46 cm |
| Standing pelvis-height reward | squared error | absolute error |

### What we had to change for MuJoCo / our robot

- **29-DoF G1.** Ours has 6 joints the paper's 23-DoF G1 doesn't (waist roll/pitch,
  wrist pitch/yaw). The policy drives the paper's 23; the 6 are held at 0 by stiff PD,
  like that robot's locked joints. (`params.POLICY_JOINTS = "all29"` lets it drive all.)
- **Body contacts with friction, and a real torso.** In Unitree's model only the feet
  touch the ground with friction, and the torso is one round capsule; lying down, the
  robot would slide and roll like a log and could never push itself up. All shapes
  have friction here, plus boxes fitted to the torso and pelvis meshes (`robot.py`).
- **PD inside MuJoCo.** HoST computes PD torques itself; at Kd = 4 on our light arm
  joints that is numerically unstable in MuJoCo. We use MuJoCo's position actuators
  with HoST's gains; motor-strength randomization scales both gains.
- **Target held per policy step.** HoST's code recomputes q + beta a every 5 ms
  substep (which makes it a constant torque); the paper's equation and the real robot
  hold it for the 20 ms step. We hold it (`params.TARGET_UPDATE`; `"substep"` = HoST).
  Worth one A/B run.
- **No restitution randomization.** MuJoCo contacts have no restitution coefficient.
- **Stairs and mixed worlds** measure heights from the ground under the pelvis (one
  downward ray), so a robot lying head-up on stairs doesn't count as standing.

### For running it on the robot later

Each saved model has `policy.onnx` (456 inputs -> 23 actions in [-1, 1]) and
`policy_meta.json` next to it: joint order, Kp/Kd, beta, the observation layout.
Per policy step: `q_target = q_measured + beta * action` with beta = 0.25 (the paper
used 0.3 on hardware for less jerk), held joints at 0, pelvis IMU. The paper raised
hip/knee Kp to 200/275 on the real G1 to close a torque gap.

## Files

| File | |
|---|---|
| `params.py` | every number, with its source |
| `robot.py` | the G1 for standing up (contacts, actuators, joints) |
| `terrains.py` | platform, wall, slope/ramp, stairs tiles and the worlds |
| `actions.py` | HoST's action, the pulling force, the settling phase |
| `mdp.py` | heights, observation, every reward, terminations |
| `events.py` | postures and resets, actuator randomization, the curriculum |
| `tasks.py` | the 9 tasks |
| `algorithm.py` | multi-critic PPO with L2C2 (port of HoST's rsl_rl) |
| `runner.py` | what train/play use: reward groups, logging, checkpoints, ONNX export |
| `scripts/` | train, play, eval (+ gallery, videos), check, check_ppo, bench |
| `cluster/` | Killarney / Slurm: setup, smoke test, training array, collect |
