#!/bin/bash
# One-time setup on a Digital Research Alliance of Canada cluster (Fir, Nibi,
# Rorqual, Trillium, Narval...). Run on a LOGIN node: compute nodes usually have
# no internet. Builds a Python venv with the same versions as docker/Dockerfile.
#
#   bash training/cluster/setup_env.sh
#
# Change the variables below once we know our cluster and allocation.
set -euo pipefail

# --- Change these --------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/humanoid}"            # where this repo is cloned (with submodules)
VENV_DIR="${VENV_DIR:-$HOME/venvs/mjlab}"         # where the Python env goes
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"     # `module spider python` to see versions
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"           # `module spider cuda`; warp/torch bring their own runtime
# ---------------------------------------------------------------------------------

module load StdEnv/2023 "$PYTHON_MODULE" "$CUDA_MODULE"

if [ ! -d "$REPO_DIR/training/unitree_rl_mjlab/src" ]; then
  echo "Missing $REPO_DIR/training/unitree_rl_mjlab. Clone with:"
  echo "  git clone --recurse-submodules -b competition https://github.com/University-of-Toronto-Humanoid-Robotics/humanoid.git $REPO_DIR"
  exit 1
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
# Same pins as docker/Dockerfile (mjlab's open ranges pull versions that break it).
# --no-index is NOT used on purpose: these come from PyPI, not the Alliance wheelhouse.
pip install mjlab==1.2.0 mujoco-warp==3.5.0 mujoco==3.5.0 warp-lang==1.12.0 \
  scipy==1.15.3 torch==2.14.1 tensorboard

# Quick check that the tasks load (no GPU needed for this).
cd "$REPO_DIR/training/scripts"
python -c "import _setup; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Comp' in t])"
echo "Done. Venv: $VENV_DIR"
