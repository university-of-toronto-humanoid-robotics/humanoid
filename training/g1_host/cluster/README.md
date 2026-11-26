# Training the G1 stand-up policy on Killarney

What this does: trains the robot's fall-recovery ("stand up") policies, one H100 per
policy, with a standard Slurm job array. No code changes or file edits needed; the
only thing you supply is your PI's allocation (`aip-<PI>`).

Everything lives in `$SCRATCH` (Killarney can't run jobs from `$HOME`). Results go to
`$SCRATCH/humanoid_runs/`.

## Before the first time

- A CCDB account on the PI's **AIP** allocation, Duo MFA set up, and an SSH key added in CCDB.
- A scratch directory: Vector creates it on request (ops-help@vectorinstitute.ai, mention the PI).
- Access to the GitHub repo `University-of-Toronto-Humanoid-Robotics/humanoid` (ask Markiyan),
  or a copy of it.

## The commands

```bash
ssh <you>@killarney.alliancecan.ca

# 1. Get the code (once; later: cd $SCRATCH/humanoid && git pull && git submodule update)
cd $SCRATCH
git clone --recurse-submodules -b marki-standup https://github.com/University-of-Toronto-Humanoid-Robotics/humanoid.git
cd humanoid

# 2. Python environment (once, ~10 min, on the login node). Ends with "SETUP OK".
bash training/g1_host/cluster/setup.sh

# 3. (Probably skippable step) A ~10-minute test on one H100. The last line of
#    host-smoke-<jobid>.out says "SMOKE PASS" or what failed. It also compiles the
#    GPU kernels that the training jobs then reuse.
sbatch --account=aip-<PI> training/g1_host/cluster/smoke.sbatch

# 4. The training: 3 policies, one after the other on a single H100, a few hours
#    to ~15 h each (about 1 GPU-day in total, probably less).
sbatch --account=aip-<PI> training/g1_host/cluster/train.sbatch
#    Not finished after 24 h? Once those jobs have ended, submit the same line
#    again: unfinished policies continue from their last checkpoint, finished
#    ones only re-run their evaluation.

# 5. Pack the results and send them to Markiyan.
bash training/g1_host/cluster/collect.sh      # prints the scp command
```

To skip typing the account: `export SBATCH_ACCOUNT=aip-<PI>` (e.g. in `~/.bashrc`).

The 3 policies:

| Policy | The robot starts... | Why |
|---|---|---|
| `HoST-Ground` | on its back on flat ground | checks our implementation against the paper (99.5% there) |
| `HoST-Supine-Mixed` | on its back, a side, or sitting against a wall; on flat ground, platforms, walls, ramps, stairs | the one the robot uses when it lands on its back |
| `HoST-Prone-Mixed` | face down or on a side; on flat ground, ramps, stairs | the one the robot uses when it lands face down |

With GPUs to spare, `--array=0-2` (in the sbatch line) runs all three at once.

## Checking on it

```bash
squeue --me                                   # running / waiting
tail -f host-<jobid>_<i>.out                  # one policy's log
grep "Learning iteration" host-<jobid>_*.out | tail -5
```

A healthy run: the `Curriculum/force_and_bound/pull_force` line falls from 200
toward 0 and `beta` from 1.0 toward 0.25 as the robots learn to stand without help;
`success_rate` rises toward 1. Ground takes 12,000 iterations, the mixed ones
20,000; the `ETA:` line (`grep ETA host-*.out | tail -1`) gives the time left.

## What each file does

| File | What it does |
|---|---|
| `setup.sh` | Makes `$SCRATCH/venvs/mjlab` with pinned versions from PyPI (same as our Docker image) and checks the tasks load |
| `env.sh` | Shared by the jobs: paths, `module load python`, the venv, kernel cache in `$SCRATCH` |
| `smoke.sbatch` | 1 H100, ~10 min (30 min limit): GPU/driver check, code checks, 5 training iterations |
| `train.sbatch` | Array of 3 jobs, one at a time, 1 H100 each (`--gres=gpu:h100:1`, 6 CPUs, 64 GB, 24 h); resumes from the last checkpoint; evaluates when done |
| `collect.sh` | One `.tar.gz` with the learning curves, last checkpoints, exported policies and evaluations |

## If something goes wrong

| Symptom | Fix |
|---|---|
| `SMOKE FAIL: torch/warp can't use the GPU` | The driver is older than torch's CUDA 13 build. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash training/g1_host/cluster/setup.sh`, then the smoke job again |
| `The repo is in $HOME` | Clone it into `$SCRATCH` instead (step 1) |
| `module: python/3.12 not found` | `module avail python`, then `PYTHON_MODULE=python/3.12.4` in front of the commands (setup and sbatch) |
| A job ran out of time | Normal for long policies: submit the same `train.sbatch` line again |
| Out of GPU memory | `NUM_ENVS=4096` (the default) or lower in front of `sbatch` |

Scratch files not used for 60 days are deleted: send the `collect.sh` tarball soon.
Questions about Killarney itself: ops-help@vectorinstitute.ai.
