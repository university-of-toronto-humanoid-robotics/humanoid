FROM ros:jazzy

RUN apt-get update && apt-get install -y ros-jazzy-desktop python3-pip && rm -rf /var/lib/apt/lists/*
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

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
