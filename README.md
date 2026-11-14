# humanoid

Clone this repo (all our code lives here), then set up for your OS. New here? Then do the [onboarding task](docs/onboarding/README.md).

**Want to walk the G1 robot right away? → [training/README.md](training/README.md)**

**Developing natively on Apple Silicon? → [docs/native-macos.md](docs/native-macos.md)**

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
  interfaces/    messages (unitree_hg: the G1's own /lowstate and /lowcmd)
  bridge/        sim ⇄ ROS 2
  teleop/        gamepad/keyboard
  policy/        walk_controller: runs a trained walking policy
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

# talking to the real G1 (not needed for the sim)
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
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

## Walking (sim and robot)

One launch file starts the whole walking stack. `sim` picks what plays the hardware, and nothing else changes:

```
hardware                                   the same software either way
  sim:=true   MuJoCo (mujoco_g1_walk_sim)    joint_state_bridge   /lowstate → /joint_states → TF
  sim:=false  the real G1 + Livox driver     perception           LiDAR → FAST-LIO → elevation map → /height_scan
                                             walk_controller      /lowstate + /height_scan + /cmd_vel → policy → /lowcmd
  out: /lowstate, /livox/lidar, /livox/imu
  in:  /lowcmd
```

`/lowstate` and `/lowcmd` are Unitree's own messages (`ros2_ws/src/interfaces/unitree_hg`): the real G1 publishes one and obeys the other.

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch bringup g1.launch.py                                                      # sim: flat ground, scenario room_walk
ros2 launch bringup g1.launch.py scenario:=room_stairs_heightmap scan:=elevation_map  # sim: stairs, on the live map
```

A scenario (`sim/scenarios/`, default `room_walk`) sets the scene, spawn pose, policy and motion script. Copy one to make a new test. The sim needs an NVIDIA GPU for elevation mapping. Details and reference results: [ros2_ws/src/README.md](ros2_ws/src/README.md).

**On the real robot this is untested.** The command will be:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp     # the G1 speaks CycloneDDS
ros2 launch bringup g1.launch.py sim:=false policy_type:=rsl_rl_g1_heightmap checkpoint:=<model.pt>
ros2 service call /walk_controller/start std_srvs/srv/Trigger    # stand, then walk. Robot on the gantry, still
ros2 service call /walk_controller/stop std_srvs/srv/Trigger     # go limp
```

Before the first try, these need doing on the robot:
- Check that ROS 2 Jazzy hears `/lowstate` from the G1 (Unitree only supports Foxy and Humble).
- Switch off Unitree's built-in controller first, or the two fight over the motors.
- Set the LiDAR's and the computer's addresses in `livox_ros_driver2/config/MID360_config.json`.
- Have a way to send `/cmd_vel` (nothing does yet: `teleop/` is empty) and a person on the emergency stop.

## LiDAR-only simulation

`mujoco_mid360_sim` moves the G1 along a scripted path with no physics, to test perception by itself. FAST-LIO's error against the true pose is printed at the end. Build with `--symlink-install`.

```bash
ros2 launch bringup perception.launch.py use_sim_time:=true
ros2 run bridge mujoco_mid360_sim                         # second terminal
```

Scans have Mid-360-like noise by default (range, direction, dropout); `--ros-args -p noise:=false` gives ideal scans.

The Docker image has the simulators' Python packages (`mujoco-lidar`, `warp-lang`, CPU `torch`). Natively, `pip install --no-deps --target .sim-deps mujoco-lidar==0.3.5 warp-lang==1.12.0` and add `.sim-deps` to `PYTHONPATH`, which leaves your other pins alone.

## Height scan (map to policy)

`height_scan_node` (`ros2_ws/src/perception/height_scan/`, started by `perception.launch.py`) turns the elevation map into the 693-point terrain scan the height-map policies were trained on, 50 times a second. `walk_controller` reads it.

| Topic | Type | What |
|---|---|---|
| `/height_scan` | `Float32MultiArray` | 693 values for the policy, in training order (see `height_scan/scan.py`). Not published until there is a map and a pelvis pose |
| `/height_scan/observed_share` | `Float32` | 0-1: share of the points the map has really observed |
| `/height_scan/feet_share` | `Float32` | 0-1: share filled from the ground under the feet (the LiDAR can't see within about 0.94 m); the rest is inpainted |
| `/height_scan/points` | `PointCloud2` | The scan as yellow dots in RViz, for checking it sits on the ground |

It needs TF for `pelvis` and the two `*_ankle_roll_link` feet, which come from `/joint_states` (made from `/lowstate` by `joint_state_bridge`). If `training/g1_comp/sensors.py` changes, change `scan.py` to match. Check with:

```bash
cd ros2_ws/src && PYTHONPATH=perception/height_scan:policy python3 -m pytest perception/height_scan/test policy/test -q
```
