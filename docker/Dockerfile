FROM ros:jazzy

RUN apt-get update && apt-get install -y ros-jazzy-desktop python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages mujoco==3.14.0

# perception: fast_lio + livox_ros_driver2
RUN apt-get update && apt-get install -y \
    git cmake build-essential \
    ros-jazzy-pcl-ros libpcl-dev libeigen3-dev libapr1-dev libaprutil1-dev \
    && rm -rf /var/lib/apt/lists/*

# Livox-SDK2, linked by livox_ros_driver2. PIC is required because the driver is a shared library.
ARG LIVOX_SDK2_COMMIT=c0796f04c143143899c87a773d9f6b7136453c0b
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 \
    && git -C /tmp/Livox-SDK2 checkout ${LIVOX_SDK2_COMMIT} \
    && cmake -S /tmp/Livox-SDK2 -B /tmp/Livox-SDK2/build -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && cmake --build /tmp/Livox-SDK2/build -j \
    && cmake --install /tmp/Livox-SDK2/build \
    && rm -rf /tmp/Livox-SDK2

# MuJoCo LiDAR simulation (ros2_ws/src/bridge). mujoco-lidar only needs numpy, mujoco and warp at runtime,
# so --no-deps stops pip replacing ROS's apt numpy. Without a GPU, warp traces rays on the CPU.
# torch is the CPU build, only for reading the walking policy's checkpoint.
RUN apt-get update && apt-get install -y python3-scipy && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages --no-cache-dir --no-deps mujoco-lidar==0.3.5 warp-lang==1.12.0 \
    && pip install --break-system-packages --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# RL training (mjlab) gets its own venv so its pinned MuJoCo doesn't clash with the one above.
# mjlab/mujoco-warp match unitree_rl_mjlab's setup.py. The rest are pinned because mjlab's
# open ranges pull versions that break it (newer mujoco/warp), and it imports scipy without declaring it.
# Activate with: source /opt/mjlab/bin/activate
RUN python3 -m venv /opt/mjlab && /opt/mjlab/bin/pip install --no-cache-dir \
    mjlab==1.2.0 mujoco-warp==3.5.0 mujoco==3.5.0 warp-lang==1.12.0 scipy==1.15.3 torch==2.14.1

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
