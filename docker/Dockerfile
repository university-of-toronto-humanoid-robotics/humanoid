FROM ros:jazzy

RUN apt-get update && apt-get install -y ros-jazzy-desktop python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages mujoco==3.14.0

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
