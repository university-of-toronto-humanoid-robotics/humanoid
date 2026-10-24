# humanoid

Clone this repo (all our code lives here), then set up for your OS. New here? Then do the [onboarding task](docs/onboarding/README.md).

## Structure

```
docker/          shared environment
docs/            onboarding, architecture
  decisions/     why we chose X
description/     URDF + MJCF (g1/ now)
sim/             MuJoCo scenes, terrains
ros2_ws/src/
  bringup/       launch files, configs
  interfaces/    custom messages
  bridge/        sim ⇄ ROS 2
  teleop/        gamepad/keyboard
  policy/        runs trained policies
  perception/    SLAM (vendored fast_lio, livox driver)
training/        RL configs (w/ controls)
tools/           scripts, motion files
```

## Windows / macOS

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - In the Docker Desktop app open Settings -> Resources -> WSL Integration -> Click the toggle button that says "Ubuntu 24.04"
2. Start the container, either:
   - **VS Code:** with the **Dev Containers** extension, open the repo folder → **Reopen in Container** → pick your OS.
   - **Terminal**, from the repo folder (`<os>` is `windows` or `mac`):
     ```bash
     docker compose -f docker/compose.<os>.yaml up -d --build   # start
     docker compose -f docker/compose.<os>.yaml exec dev bash   # open a shell (repeat for more)
     docker compose -f docker/compose.<os>.yaml down            # stop
     ```

GUI windows open on your desktop (Windows) or at <http://localhost:8080/vnc.html> (Mac).

## Ubuntu 24.04

With [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) (`ros-jazzy-desktop`) installed:

```bash
pip install --break-system-packages mujoco==3.14.0
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# perception deps (fast_lio, livox_ros_driver2)
sudo apt install ros-jazzy-pcl-ros libapr1-dev libaprutil1-dev
git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 && cd /tmp/Livox-SDK2
git checkout c0796f04c143143899c87a773d9f6b7136453c0b
cmake -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON && cmake --build build -j
sudo cmake --install build && cd - && rm -rf /tmp/Livox-SDK2
```

Keep your machine in sync with `docker/Dockerfile`.

## Check it works

```bash
ros2 run demo_nodes_cpp talker   # Ctrl+C to stop
python3 -m mujoco.viewer         # a MuJoCo window opens
```
