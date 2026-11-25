#!/usr/bin/env bash
# Install this repository's ROS simulation stack directly in a hosted Colab GPU VM.
# Run as root on Ubuntu 24.04 x86_64. All changes are in the disposable runtime.
set -euo pipefail

repo_dir=${1:?Pass the cloned repository directory}
if [[ $(id -u) != 0 || $(uname -m) != x86_64 ]]; then
  echo 'ROS Colab setup requires a root, x86_64 Linux runtime.' >&2
  exit 1
fi
source /etc/os-release
if [[ ${ID} != ubuntu || ${VERSION_ID} != 24.04 ]]; then
  echo "ROS Jazzy apt packages require Ubuntu 24.04; found ${PRETTY_NAME}." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name --format=csv,noheader; then
  echo 'Select a hosted Colab NVIDIA GPU runtime before installing ROS.' >&2
  exit 1
fi

echo '[1/5] Installing Ubuntu prerequisites and Python 3.12 for ROS Jazzy.'
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3.12 python3-pip curl ca-certificates gnupg software-properties-common \
  git cmake build-essential libpcl-dev libeigen3-dev libapr1-dev libaprutil1-dev \
  python3-opencv python3-scipy python3-shapely python3-ruamel.yaml python3-yaml

# Hosted Colab may point /usr/bin/python3 at its Python 3.13 kernel. Jazzy's
# Ubuntu binaries and apt Python extensions target 3.12. Changing the link in
# this disposable VM does not change the running notebook kernel.
if [[ $(/usr/bin/python3 -c 'import sys; print(sys.version_info.minor)') != 12 ]]; then
  echo 'Selecting Ubuntu Python 3.12 for ROS executables.'
  ln -sfn /usr/bin/python3.12 /usr/bin/python3
fi
/usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
# ROS executable shebangs use /usr/bin/env python3. Colab puts its own Python
# ahead of /usr/bin, so prioritize Ubuntu's 3.12 in child shells.
export PATH=/usr/bin:/bin:${PATH}

echo '[2/5] Adding the ROS 2 Jazzy apt repository and installing ROS packages.'
add-apt-repository -y universe >/dev/null
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main\n' \
  > /etc/apt/sources.list.d/ros2.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ros-jazzy-desktop ros-jazzy-pcl-ros ros-jazzy-grid-map-msgs \
  ros-jazzy-grid-map-rviz-plugin ros-jazzy-tf-transformations \
  ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-image-transport-plugins \
  ros-jazzy-rosbag2 python3-colcon-common-extensions

echo '[3/5] Building the pinned Livox SDK.'
if [[ ! -f /content/.humanoid_livox_sdk_ready ]]; then
  sdk_dir=/content/humanoid-livox-sdk
  if [[ ! -d ${sdk_dir}/.git ]]; then
    git clone -q https://github.com/Livox-SDK/Livox-SDK2.git "${sdk_dir}"
  fi
  git -C "${sdk_dir}" checkout -q c0796f04c143143899c87a773d9f6b7136453c0b
  cmake -S "${sdk_dir}" -B "${sdk_dir}/build" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build "${sdk_dir}/build" -j 2
  cmake --install "${sdk_dir}/build"
  ldconfig
  touch /content/.humanoid_livox_sdk_ready
fi

echo '[4/5] Installing pinned simulation and CUDA Python packages for ROS Python 3.12.'
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir mujoco==3.14.0
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
  mujoco-lidar==0.3.5 warp-lang==1.12.0
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
# Colab's /usr/local/cuda currently points to CUDA 13.0 headers. Use the
# matching 12.9 NVRTC and runtime-wheel headers for this CuPy 12.x install.
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
  cupy-cuda12x==13.6.0 nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cuda-runtime-cu12==12.9.79
# CuPy's wheel imports fastrlock; --no-deps above intentionally skips it.
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir 'fastrlock>=0.5'
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir simple-parsing==0.1.9

echo '[5/5] Building this repository ROS workspace.'
# ROS's generated setup scripts read optional variables without guarding them.
# Temporarily disable nounset while sourcing them.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd "${repo_dir}/ros2_ws"
colcon build --symlink-install --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
set +u
source install/setup.bash
set -u
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export CUDA_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime
test -f "${CUDA_PATH}/include/cuda_fp8.hpp"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=42
/usr/bin/python3 -c 'import rclpy, mujoco, mujoco_lidar, warp, cupy, cv2, scipy; print("ROS Python imports OK; CuPy devices:", cupy.cuda.runtime.getDeviceCount()); print("CuPy kernel OK:", cupy.arange(3).sum().item())'
/usr/bin/python3 - <<'PY'
import cupy as cp
from cupyx.scipy import ndimage

image = cp.zeros((8, 8), dtype=cp.float32)
ndimage.uniform_filter(image, size=3).sum().item()
print('CuPy ndimage filter compiled successfully.', flush=True)
PY
ros2 pkg prefix bringup
echo 'Running the repository CuPy map-update smoke test.'
/usr/bin/python3 - <<'PY'
import runpy

checks = runpy.run_path(
  'src/perception/elevation_mapping_cupy/elevation_mapping_cupy/'
  'elevation_mapping_cupy/tests/test_kernel_compile_smoke.py'
)
checks['test_cupy_cuda_is_available']()
checks['test_kernels_compile_and_one_update_step_runs']()
print('CuPy map update passed.', flush=True)
PY
echo 'Checking the ROS elevation mapper with the same parameter files as bringup.'
/usr/bin/python3 - <<'PY'
import os
import rclpy
from ament_index_python.packages import get_package_share_directory
from elevation_mapping_cupy.elevation_mapping_node import ElevationMappingNode

core = os.path.join(get_package_share_directory('elevation_mapping_cupy'),
                    'config', 'core', 'core_param.yaml')
g1 = os.path.join(get_package_share_directory('bringup'),
                  'config', 'g1_elevation_mapping.yaml')
rclpy.init(args=['--ros-args', '--params-file', core, '--params-file', g1])
node = None
try:
  node = ElevationMappingNode()
  print('ROS elevation mapper initialized successfully.', flush=True)
finally:
  if node is not None:
    node.destroy_node()
  rclpy.try_shutdown()
PY
echo 'ROS simulation setup is ready.'
