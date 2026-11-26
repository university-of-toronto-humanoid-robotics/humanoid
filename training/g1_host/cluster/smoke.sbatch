#!/bin/bash
# A quick test on one H100 before the real runs (~10 min). From the repo in $SCRATCH:
#
#   sbatch --account=aip-<PI> training/g1_host/cluster/smoke.sbatch
#
# Checks that the GPU, the Python environment and our code work, and that a short
# training run writes a checkpoint. The last line of host-smoke-<jobid>.out is
# "SMOKE PASS" or "SMOKE FAIL: <which step>". The first run also compiles the GPU
# kernels once, so the training jobs start faster.
#SBATCH --job-name=host-smoke
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
set -uo pipefail

# Paths and the Python environment. sbatch runs a copy of this file, so env.sh is
# found from the folder the job was submitted from (the repo).
source "${SLURM_SUBMIT_DIR:-$PWD}/training/g1_host/cluster/env.sh"

fail() { echo "SMOKE FAIL: $1"; exit 1; }

echo "=== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail "nvidia-smi"
python - <<'PY' || fail "torch/warp can't use the GPU: rerun setup.sh with TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
import torch, warp
assert torch.cuda.is_available(), "torch sees no GPU"
print("torch", torch.__version__, "CUDA", torch.version.cuda, "on", torch.cuda.get_device_name(0))
warp.init()
PY

echo "=== Code checks"
cd "$REPO_DIR/training"
python g1_host/scripts/check_ppo.py || fail "check_ppo.py"
python g1_host/scripts/check.py HoST-Ground --num-envs 64 || fail "check.py"

echo "=== Training: 5 iterations, 4096 robots"
cd "$RUN_DIR"
python "$REPO_DIR/training/g1_host/scripts/train.py" HoST-Ground --env.scene.num-envs 4096 \
  --agent.max-iterations 5 --agent.run-name smoke --agent.logger tensorboard || fail "train.py"

echo
echo "SMOKE PASS"
