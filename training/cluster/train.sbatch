#!/bin/bash
# Train a G1-Comp task on one GPU with Slurm (Digital Research Alliance of Canada).
#
#   sbatch training/cluster/train.sbatch                          # full run
#   SMOKE=1 sbatch --time=0:30:00 training/cluster/train.sbatch   # 30-iteration test first
#   TASK=G1-Comp-Course ITERATIONS=3000 sbatch training/cluster/train.sbatch
#
# Watch: squeue -u $USER ; tail -f slurm-<jobid>.out
# Results: $RUN_DIR/logs/rsl_rl/<experiment>/<date>_<run name>/model_<N>.pt
#
# --- Change these #SBATCH lines for our allocation ------------------------------
#SBATCH --account=def-CHANGEME          # our PI's allocation, e.g. def-smithj or rrg-smithj
#SBATCH --gpus-per-node=h100:1          # one H100 (Narval has a100); multi-GPU won't see our tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                 # jobs <= 24 h usually start sooner
#SBATCH --job-name=g1-comp
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------------
set -euo pipefail

# --- Change these (or pass them as environment variables) -----------------------
REPO_DIR="${REPO_DIR:-$HOME/humanoid}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/mjlab}"
RUN_DIR="${RUN_DIR:-$SCRATCH/humanoid_runs}"      # checkpoints and logs (scratch: big, not backed up)
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
TASK="${TASK:-G1-Comp-Rough}"
NUM_ENVS="${NUM_ENVS:-4096}"                      # 4096 fits an H100 easily; try 8192 later
ITERATIONS="${ITERATIONS:-10000}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-local}}"
RESUME_RUN="${RESUME_RUN:-}"                      # a run folder name to continue from, or empty
SMOKE="${SMOKE:-0}"                               # 1 = 256 robots, 30 iterations
# Crouching, arm override and extra standing robots ramp in between these iterations
# (after the policy walks). Use 0 and 0 to have them on from the start.
export G1_COMP_SKILLS_START="${SKILLS_START:-500}"
export G1_COMP_SKILLS_FULL="${SKILLS_FULL:-2000}"
# ---------------------------------------------------------------------------------

module load StdEnv/2023 "$PYTHON_MODULE" "$CUDA_MODULE"
source "$VENV_DIR/bin/activate"
export WARP_CACHE_PATH="$SCRATCH/.cache/warp"     # compiled GPU kernels; home quota is small
export MUJOCO_GL=egl                              # headless rendering
export WANDB_MODE=offline                         # compute nodes have no internet

if [ "$SMOKE" = "1" ]; then
  NUM_ENVS=256
  ITERATIONS=30
fi

mkdir -p "$RUN_DIR" "$WARP_CACHE_PATH"
cd "$RUN_DIR"                                     # train.py writes ./logs/rsl_rl/... here
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS=(
  "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$ITERATIONS"
  --agent.logger tensorboard
  --agent.run-name "$RUN_NAME"
)
if [ -n "$RESUME_RUN" ]; then
  ARGS+=(--agent.resume True --agent.load-run "$RESUME_RUN")
fi

echo "Training $TASK with $NUM_ENVS robots for $ITERATIONS iterations in $RUN_DIR"
python "$REPO_DIR/training/scripts/train.py" "${ARGS[@]}"
