# humanoid

Clone this repo (all our code lives here), then set up for your OS. New here? Then do the [onboarding task](docs/onboarding/README.md).

**Want to walk the G1 robot right away? → [training/README.md](training/README.md)**

## Structure

```
docker/          shared environment
docs/            onboarding, architecture
  decisions/     why we chose X
description/     URDF + MJCF (g1/ now)
sim/             MuJoCo scenes, terrains
  competition/   the competition course, built from the organizers' CAD
ros2_ws/src/
  bringup/       launch files, configs
  interfaces/    custom messages
  bridge/        sim ⇄ ROS 2
  teleop/        gamepad/keyboard
  policy/        runs trained policies
  perception/    SLAM (vendored fast_lio, livox driver)
training/        RL: Unitree's code (submodule) + our trained models
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

# elevation mapping (elevation_mapping_cupy) — NVIDIA GPU only
sudo apt install ros-jazzy-grid-map-msgs ros-jazzy-grid-map-rviz-plugin ros-jazzy-tf-transformations python3-shapely python3-ruamel.yaml python3-opencv
pip install --user --break-system-packages cupy-cuda12x==13.6.0 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 simple-parsing==0.1.9   # cupy 14 pulls numpy 2, which breaks Ubuntu's scipy; no PyTorch needed
echo 'export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

Keep your machine in sync with `docker/Dockerfile`.

## Check it works

```bash
ros2 run demo_nodes_cpp talker   # Ctrl+C to stop
python3 -m mujoco.viewer         # a MuJoCo window opens
```

## Perception (FAST-LIO + elevation mapping)

```bash
cd ros2_ws && colcon build && source install/setup.bash
ros2 launch bringup perception.launch.py                  # on the G1
ros2 launch bringup perception.launch.py use_sim_time:=true lidar_upside_down:=false   # with an upright-LiDAR bag
ros2 bag play <bag> --clock                               # second terminal
```

Elevation mapping needs an NVIDIA GPU, so it doesn't run on the Mac container. The launch file starts FAST-LIO, not the LiDAR driver: on the G1, also run `ros2 launch livox_ros_driver2 msg_MID360_launch.py`.

## LiDAR simulation (MuJoCo + simulated Mid-360)

The G1 follows a scripted path (no physics or walking yet) and FAST-LIO's error against the true pose is printed at the end. Elevation mapping needs an NVIDIA GPU. Build with `--symlink-install`, or pass `-p scene:=<path to sim/g1_lidar_scene.xml>`.

The Docker image has the simulator's Python packages (`mujoco-lidar`, `warp-lang`, CPU `torch`). Natively, `pip install --no-deps --target .sim-deps mujoco-lidar==0.3.5 warp-lang==1.12.0` and add `.sim-deps` to `PYTHONPATH`, which leaves your other pins alone.

```bash
ros2 launch bringup perception.launch.py use_sim_time:=true
ros2 run bridge mujoco_mid360_sim                         # second terminal
ros2 run bridge mujoco_g1_walk_sim                        # or: physics, walking under training/models/g1_walk/model_12400.pt
ros2 run bridge mujoco_g1_walk_sim --ros-args -p scenario:=course_ramps   # the competition course's ramps
```

`mujoco_g1_walk_sim` runs a scenario from `sim/scenarios/` (default `room_walk`): the scene, spawn pose, policy and motion script (waypoints or timed velocity commands). Copy one to make a new test.

Scans have Mid-360-like noise by default (range, direction, dropout); `--ros-args -p noise:=false` gives ideal scans.

## Height scan (map to policy)

`height_scan_node` (`ros2_ws/src/perception/height_scan/`, started by `perception.launch.py`) turns the elevation map into the 693-point terrain scan the height-map policies were trained on, 50 times a second. The walk sim (`-p scan:=elevation_map`) reads it. Nothing on the real robot reads it yet: Unitree's `g1_ctrl` doesn't subscribe to ROS topics.

| Topic | Type | What |
|---|---|---|
| `/height_scan` | `Float32MultiArray` | 693 values for the policy, in training order (see `height_scan/scan.py`). Not published until there is a map and a pelvis pose |
| `/height_scan/observed_share` | `Float32` | 0-1: share of the points the map has really observed |
| `/height_scan/feet_share` | `Float32` | 0-1: share filled from the ground under the feet (the LiDAR can't see within about 0.94 m); the rest is inpainted |
| `/height_scan/points` | `PointCloud2` | The scan as yellow dots in RViz, for checking it sits on the ground |

It needs TF for `pelvis` and the two `*_ankle_roll_link` feet, which come from `/joint_states`. If `training/g1_comp/sensors.py` changes, change `scan.py` to match. Check with:

```bash
cd ros2_ws/src/perception/height_scan && python3 -m pytest test -q
```
