FROM ros:jazzy

RUN apt-get update && apt-get install -y ros-jazzy-desktop python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages mujoco==3.14.0

# RL training (mjlab) gets its own venv so its pinned MuJoCo doesn't clash with the one above.
# mjlab/mujoco-warp match unitree_rl_mjlab's setup.py. The rest are pinned because mjlab's
# open ranges pull versions that break it (newer mujoco/warp), and it imports scipy without declaring it.
# Activate with: source /opt/mjlab/bin/activate
RUN python3 -m venv /opt/mjlab && /opt/mjlab/bin/pip install --no-cache-dir \
    mjlab==1.2.0 mujoco-warp==3.5.0 mujoco==3.5.0 warp-lang==1.12.0 scipy==1.15.3 torch==2.14.1

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
