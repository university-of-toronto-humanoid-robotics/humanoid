# Four-expert student distillation

Run from `training/` with the mjlab Python environment, or use
`notebooks/train_student.ipynb` on Colab. The expert PPO tasks are unchanged.

The frozen ascend, descend, ramps, and obstacles experts see their original
791-value, clean playback observations. The student sees the ROS-like current
elevation map, its previous 3 Hz map aligned to the current pelvis, five joint
position frames, and the remaining G1 state. Student joint positions receive
independent Gaussian noise (0.01 rad per new frame), and current joint velocities
receive 0.5 rad/s Gaussian noise. The 1600-value student input produces 29 joint
actions through a two-channel CNN and the existing 512–256–256–128 MLP design.

## BC warmup

```bash
python scripts/train_student_bc.py \
  --teacher-ascend models/RPL/ascend/model_v5_1998_best.pt \
  --teacher-descend models/RPL/descend/model_v2_3496_best.pt \
  --teacher-ramps models/RPL/ramps/model_v2_3498_best.pt \
  --teacher-obstacles models/RPL/obstacles/model_1998.pt \
  --num-envs 1024 --updates 1000 --run-name bc_1
```

`--num-envs` is the total across four equal pools and must be divisible by four.
BC needs at least 40 so every pool can initially cover all ten levels. It
randomizes levels on reset and samples equally across terrains and levels.
The experts control the robots; the student learns from their deterministic
action labels. Each save writes `student_<update>.pt` (weights, optimizer, and
resume state) and `demos_<update>.pt` (observation/action replay) under
`training/logs/student_distill/bc/<run>/checkpoints/`.

## DAgger

```bash
python scripts/train_student_dagger.py \
  --teacher-ascend models/RPL/ascend/model_v5_1998_best.pt \
  --teacher-descend models/RPL/descend/model_v2_3496_best.pt \
  --teacher-ramps models/RPL/ramps/model_v2_3498_best.pt \
  --teacher-obstacles models/RPL/obstacles/model_1998.pt \
  --init-mode bc-with-demos \
  --bc-checkpoint logs/student_distill/bc/<run>/checkpoints/student_1000.pt \
  --bc-demo-checkpoint logs/student_distill/bc/<run>/checkpoints/demos_1000.pt \
  --num-envs 1024 --updates 2000 --run-name dagger_1
```

Choose one start mode: `bc-with-demos`, `bc-weights`, or `scratch`. The first
loads BC weights and balanced demo replay. You may choose a demo file from a
different BC save. If `--bc-demo-checkpoint` is omitted, the trainer loads the
demo file paired with `--bc-checkpoint`. Older combined checkpoints remain
loadable. Its BC share falls from 50% to 10%
over 1000 DAgger updates. The second loads weights but starts with no demos.
The third starts from random weights. In all three modes the **student** controls
the robots, while each frozen expert labels the state visited by its terrain
pool. The four curricula progress independently based on student traversal.
The latter two modes train only on student-visited examples.

## Colab notebook

Run `notebooks/train_student.ipynb` in order. Its first cell can authenticate
GitHub and clone the selected branch, or load `humanoid-source.tar.gz` from
Drive when GitHub login is unavailable. Then it installs the mjlab dependencies
and mounts Drive. In the options cell choose BC, one of the three DAgger start modes,
or resume; choose the four expert checkpoints and the student weight and demo
checkpoints separately. New saves keep weights and demos in separate `.pt` files.
Checkpoints and logs go directly to `MyDrive/humanoid/logs/student_distill`.
Rerun the options cell after BC finishes to discover its new checkpoint. For
GitHub mode, push this notebook and the trainer code to the selected branch
before running Colab. For archive mode, recreate the archive from the latest
local code.

Both scripts accept `--rollout-steps`, `--batch-size`, `--gradient-steps`,
`--learning-rate`, `--replay-capacity` (per terrain per source), `--save-every`,
`--device`, `--logger tensorboard|wandb|none`, `--wandb-project`, and `--run-dir`.
Default logging is TensorBoard. W&B uses the usual `WANDB_API_KEY` environment
variable. `--resume <student checkpoint>` restores weights, optimizer, random
state, and each environment's level assignments from the student file, plus
replay from the matching demos file; the simulation restarts from
fresh episodes. Use the same expert checkpoints, number of environments, and
replay capacity when resuming. Increase `--updates` beyond the saved update.

**Start with a small GPU run.** Four terrain worlds, the student map, and replay
all add memory cost. Check the loss and per-terrain level and success metrics
before increasing parallel environments. The training loss imitates actions;
environment rewards are not optimized by the student.
