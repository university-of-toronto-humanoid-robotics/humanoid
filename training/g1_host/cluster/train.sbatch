#!/bin/bash
# Train the G1 stand-up policies with a job array, one H100 per policy. From the
# repo in $SCRATCH:
#
#   sbatch --account=aip-<PI> training/g1_host/cluster/train.sbatch
#
# Default: 3 policies, one after the other (--array=0-2%1: never more than 1 H100):
#   HoST-Ground        on its back on flat ground: checks our port against the paper
#   HoST-Supine-Mixed  on its back / side / sitting, on every surface: for the robot
#   HoST-Prone-Mixed   face down / side, on every surface: for the robot
# Expected: a few hours to ~15 h each on an H100. The "ETA:" line in the log says.
# To run all three at once instead: add --array=0-2 to the sbatch line.
#
# A job stops after 24 h. Submitting the same line again (once the jobs have ended)
# continues each unfinished policy from its last checkpoint and skips finished ones.
# When a policy is done it is evaluated; the numbers go to $RUN_DIR/logs/eval/.
#SBATCH --job-name=host
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

# --- What to train (or set these in front of sbatch) ---------------------------------
TASKS="${TASKS:-HoST-Ground HoST-Supine-Mixed HoST-Prone-Mixed}"
NUM_ENVS="${NUM_ENVS:-4096}"        # robots per GPU (the paper's number)
RUN_TAG="${RUN_TAG:-killarney}"     # name of the runs; a new name starts them over
ITERATIONS="${ITERATIONS:-}"        # total iterations; empty = each policy's own (12000-20000)
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
