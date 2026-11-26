# Shared by the sbatch scripts: where things are and the Python environment.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # the repo (this file is in it)
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/mjlab}"          # made by setup.sh
RUN_DIR="${RUN_DIR:-$SCRATCH/humanoid_runs}"          # checkpoints and logs go to $RUN_DIR/logs/
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"

set +u  # Lmod and the venv's activate script read unset variables
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
set -u
export WARP_CACHE_PATH="$SCRATCH/.cache/warp"         # compiled GPU kernels: the first job compiles, later ones reuse
export MUJOCO_GL=egl                                  # headless rendering (videos)
export WANDB_MODE=offline
mkdir -p "$RUN_DIR" "$WARP_CACHE_PATH"
