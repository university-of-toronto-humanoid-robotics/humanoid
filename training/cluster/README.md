# Training on the cluster (Slurm)

Written for the Digital Research Alliance of Canada clusters, for example Fir, Nibi, Rorqual (H100), Trillium or Narval (A100). **Untested so far:** we don't have the account details yet. Every guess is a variable at the top of the two scripts, marked "Change these".

## What to find out first

| Question | Where it goes |
|---|---|
| Which cluster, and does it have H100s? | `--gpus-per-node=h100:1` in `train.sbatch` (`a100:1` on Narval) |
| Our allocation (the PI's account) | `--account=def-...` in `train.sbatch` |
| Python and CUDA module names | `PYTHON_MODULE`, `CUDA_MODULE` (check with `module spider python` and `module spider cuda`) |
| Max job length | `--time` (jobs of 24 h or less usually start sooner) |

## Steps

On a **login node** (it has internet; compute nodes usually don't):

```bash
git clone --recurse-submodules -b competition https://github.com/University-of-Toronto-Humanoid-Robotics/humanoid.git ~/humanoid
bash ~/humanoid/training/cluster/setup_env.sh     # once: builds ~/venvs/mjlab, prints our task names
```

Then submit. Start with a 30-iteration test job, then the real run:

```bash
cd ~/humanoid
SMOKE=1 sbatch --time=0:30:00 training/cluster/train.sbatch
sbatch training/cluster/train.sbatch                                 # G1-Comp-Rough, 4096 robots, 10k iterations
TASK=G1-Comp-Course ITERATIONS=3000 sbatch training/cluster/train.sbatch
RESUME_RUN=<run folder> sbatch training/cluster/train.sbatch          # continue a run
SKILLS_START=300 SKILLS_FULL=1500 sbatch training/cluster/train.sbatch # bring crouch/arms in sooner
```

- **Check:** `squeue -u $USER` shows the job, and `tail -f g1-comp-<jobid>.out` shows its output.
- **Results:** checkpoints go to `$SCRATCH/humanoid_runs/logs/rsl_rl/<experiment>/<date>_<jobid>/` every 100 iterations.
- **Curves:** copy the folder home to view the TensorBoard curves.
- **Trying a policy:** copy a `model_<N>.pt` to your laptop and use `training/scripts/play.py` or `eval_policy.py`.

## Notes

- **One GPU per job.** With several GPUs (`--gpu-ids`), mjlab starts worker processes that don't see our tasks. One H100 with 4096 robots is the normal setup anyway.
- **Same versions as the Docker image.** `setup_env.sh` pins the same versions as `docker/Dockerfile`, installed from PyPI rather than the Alliance wheelhouse. If a cluster blocks PyPI, the fallback is an Apptainer image built from our Dockerfile.
- **Scratch storage.** `$SCRATCH` is large but not backed up, and old files get purged. Copy good checkpoints to `$PROJECT` or home.
