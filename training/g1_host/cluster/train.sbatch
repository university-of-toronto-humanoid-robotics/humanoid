#!/bin/bash
# Train the G1 stand-up policies: one H100 per policy, all at once (a job array).
# From the repo in $SCRATCH:
#
#   sbatch --account=aip-<PI> training/g1_host/cluster/train.sbatch
#
# Default: the 5 policies of the paper (array 0-4). Other policies, e.g.:
#   TASKS="HoST-Supine-Mixed HoST-Prone-Mixed" sbatch --account=aip-<PI> --array=0-1 training/g1_host/cluster/train.sbatch
#
# A job stops after 24 h. Submitting the same line again (once the jobs have ended)
# continues each unfinished policy from its last checkpoint and skips finished ones.
# When a policy is done it is evaluated; the numbers go to $RUN_DIR/logs/eval/.
#SBATCH --job-name=host
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

# --- What to train (or set these in front of sbatch) ---------------------------------
TASKS="${TASKS:-HoST-Ground HoST-Prone HoST-Platform HoST-Wall HoST-Slope}"
NUM_ENVS="${NUM_ENVS:-4096}"        # robots per GPU (the smoke job's benchmark suggests a number)
RUN_TAG="${RUN_TAG:-killarney}"     # name of the runs; a new name starts them over
ITERATIONS="${ITERATIONS:-}"        # total iterations; empty = each policy's own (7500-12000)
# -------------------------------------------------------------------------------------

# Paths and the Python environment. sbatch runs a copy of this file, so env.sh is
# found from the folder the job was submitted from (the repo).
source "${SLURM_SUBMIT_DIR:-$PWD}/training/g1_host/cluster/env.sh"

TASK_LIST=($TASKS)
TASK="${TASK_LIST[$SLURM_ARRAY_TASK_ID]}"
echo "Policy $TASK on $(hostname), $NUM_ENVS robots"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cd "$RUN_DIR"   # checkpoints and logs go to $RUN_DIR/logs/
python "$REPO_DIR/training/g1_host/scripts/train.py" "$TASK" --continue "$RUN_TAG" \
  --env.scene.num-envs "$NUM_ENVS" --agent.logger tensorboard ${ITERATIONS:+--agent.max-iterations "$ITERATIONS"}
python "$REPO_DIR/training/g1_host/scripts/eval.py" --task "$TASK" --run "$RUN_TAG"
