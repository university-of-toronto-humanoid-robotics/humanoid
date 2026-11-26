#!/bin/bash
# One-time setup on Killarney (or any Alliance cluster). Run on a LOGIN node, from
# the repo, which must be under $SCRATCH (Killarney can't run jobs from $HOME):
#
#   cd $SCRATCH/humanoid && bash training/g1_host/cluster/setup.sh
#
# Builds a Python venv in $SCRATCH with the same versions as our Docker image,
# then checks that the HoST tasks load. Takes ~10 minutes. Ends with "SETUP OK".
set -eo pipefail

# --- Change only if needed ---------------------------------------------------------
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"        # `module avail python` lists them
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/mjlab}"
# torch 2.14.1 from PyPI is built for CUDA 13 (NVIDIA driver >= 580). If the smoke
# job says the GPU can't be used, rerun with a CUDA 12 build:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash training/g1_host/cluster/setup.sh
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
# ------------------------------------------------------------------------------------

if [ -z "$SCRATCH" ]; then
  echo "\$SCRATCH is not set: run this on the cluster's login node."
  exit 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
case "$REPO_DIR" in
  "$(cd "$HOME" && pwd -P)"/*)
    echo "The repo is in \$HOME ($REPO_DIR). Killarney can't run jobs from there: clone it into \$SCRATCH."
    exit 1;;
esac
if [ ! -d "$REPO_DIR/training/unitree_rl_mjlab/src" ]; then
  echo "Unitree's code is missing. In $REPO_DIR run: git submodule update --init"
  exit 1
fi

module load "$PYTHON_MODULE"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
export PIP_CONFIG_FILE=/dev/null   # plain PyPI, not the Alliance wheelhouse (it lacks mjlab and mujoco-warp)
pip install --upgrade pip
if [ -n "$TORCH_INDEX_URL" ]; then
  pip install torch==2.14.1 --index-url "$TORCH_INDEX_URL"
fi
# The same versions as docker/Dockerfile and our training container.
pip install mjlab==1.2.0 mujoco-warp==3.5.0 mujoco==3.5.0 warp-lang==1.12.0 scipy==1.15.3 torch==2.14.1 \
  rsl-rl-lib==5.0.1 tensordict==0.15.6 onnx==1.23.2 tensorboard "imageio[ffmpeg]"

cd "$REPO_DIR/training"
python - <<'PY'
import sys
sys.path[:0] = ["unitree_rl_mjlab", "."]
import g1_host  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import list_tasks
print("HoST tasks:", [t for t in list_tasks() if t.startswith("HoST")])
PY
echo "SETUP OK (venv: $VENV_DIR). Next: sbatch --account=aip-<PI> training/g1_host/cluster/smoke.sbatch"
