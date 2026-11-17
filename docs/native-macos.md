# Native macOS development

This is the native Apple Silicon workflow for inspecting and validating the
MuJoCo competition environment without Docker. Docker support remains the
team-wide Linux/Windows workflow and is unchanged.

## Scope

The M3 Pro workflow is intended for:

- viewing and checking the competition course;
- inspecting terrain, observations, rewards, and configuration code;
- small CPU-only environment checks and policy evaluation where supported.

Large-scale PPO training remains a GPU workflow. Run it on a Linux/NVIDIA
machine or Google Colab and keep the existing training scripts unchanged.
The repository's ROS 2/perception workspace is not part of this native macOS
setup.

## One-time setup

Run the following from the repository root on the Mac host:

```bash
git submodule update --init training/unitree_rl_mjlab

python3 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docs/requirements-native-mujoco.txt
```

The submodule supplies the G1 XML and mesh assets used by the robot-on-course
viewer. The native MuJoCo environment is separate from the mjlab environment
because the Docker image deliberately pins different MuJoCo versions for those
two use cases.

## Course viewer and checks

Activate the native environment whenever you work on the standalone course:

```bash
source .venv-native/bin/activate
python sim/competition/view_course.py
```

Useful checks:

```bash
python sim/competition/view_course.py --no-robot
python sim/competition/view_course.py --spawn landing
python sim/competition/view_course.py --spawn stairs --render /tmp/course-stairs.png
python tools/cad/check_course.py
```

The native viewer opens its own macOS window; no VNC page or Docker display
service is involved.

## mjlab evaluation environment

The training task code is a separate environment matching the versions pinned
in `docker/Dockerfile`:

```bash
python3 -m venv .venv-mjlab
source .venv-mjlab/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docs/requirements-native-mjlab.txt
```

Keep the local compiler and plotting caches inside the checkout so they do not
depend on macOS user-cache permissions:

```bash
mkdir -p .cache/warp .cache/matplotlib .cache/fontconfig
export WARP_CACHE_PATH="$PWD/.cache/warp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
```

Set those variables again in each shell where you run mjlab, or add them to a
local shell profile if this becomes your normal workflow.

From the `training` directory, start with a very small CPU validation:

```bash
cd training
python scripts/check_task.py G1-Comp-Course --num-envs 1 --steps 20
```

The project scripts select CUDA when it is available and otherwise select the
CPU. On macOS, mjlab is for evaluation and development checks only; do not use
this environment for the large parallel PPO runs intended for Colab.

For mjlab's native policy viewer on macOS, use MuJoCo's `mjpython` launcher
instead of `python` because the passive viewer must run on the main thread:

```bash
mjpython scripts/play.py Unitree-G1-Flat \
  --viewer native \
  --num-envs 1 \
  --device cpu \
  --checkpoint-file models/g1_walk/model_12400.pt
```

## Returning to Docker

Nothing in this workflow removes or changes Docker support. To use the team
container again, deactivate the Python environment and follow the root
README's Docker commands.
