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

# 3. A one-hour test on one H100. The last line of host-smoke-<jobid>.out says
#    "SMOKE PASS" or what failed; it also prints how many robots per GPU to use,
#    and compiles the GPU kernels that the training jobs then reuse.
sbatch --account=aip-<PI> training/g1_host/cluster/smoke.sbatch

# 4. The training: 5 policies, 5 H100s at once, up to 24 h per job.
sbatch --account=aip-<PI> training/g1_host/cluster/train.sbatch
#    Not finished after 24 h? Once those jobs have ended, submit the same line
#    again: unfinished policies continue from their last checkpoint, finished
#    ones only re-run their evaluation.

# 5. Pack the results and send them to Markiyan.
bash training/g1_host/cluster/collect.sh      # prints the scp command
```

To skip typing the account: `export SBATCH_ACCOUNT=aip-<PI>` (e.g. in `~/.bashrc`).

Later runs use the same `train.sbatch` with other policies, for example the two
mixed-terrain ones:

```bash
TASKS="HoST-Supine-Mixed HoST-Prone-Mixed" sbatch --account=aip-<PI> --array=0-1 training/g1_host/cluster/train.sbatch
```

and, if the smoke job recommended more robots per GPU, `NUM_ENVS=8192` in front.

## Checking on it

```bash
squeue --me                                   # running / waiting
tail -f host-<jobid>_<i>.out                  # one policy's log
grep "Learning iteration" host-<jobid>_*.out | tail -5
```

A healthy run: the `Curriculum/force_and_bound/pull_force` line falls from 200
toward 0 and `beta` from 1.0 toward 0.25 as the robots learn to stand without help;
`success_rate` rises toward 1. Each policy takes about 7,500-12,000 iterations.

## What each file does

| File | What it does |
|---|---|
| `setup.sh` | Makes `$SCRATCH/venvs/mjlab` with pinned versions from PyPI (same as our Docker image) and checks the tasks load |
| `env.sh` | Shared by the jobs: paths, `module load python`, the venv, kernel cache in `$SCRATCH` |
| `smoke.sbatch` | 1 H100, 1 h: GPU/driver check, code checks, 20 training iterations, robots-per-GPU benchmark |
| `train.sbatch` | Array job, 1 H100 per policy (`--gres=gpu:h100:1`, 6 CPUs, 64 GB, 24 h); resumes from the last checkpoint; evaluates when done |
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
