# teleop

The operator station: drive the G1 from a keyboard or a PS4 controller, using
only what the robot sees and reports.

Competition rule: the operator sits at a separate station and may not look at
the robot during a trial. Everything they need comes through this interface.

This README is also the user guide for testing it in simulation: set up once,
then open three terminals and pick a test from [Tests](#new-tests).

## How it works

teleop is the `/cmd_vel` source in the stack described in
[`ros2_ws/src/README.md`](../README.md). Everything below the operator is the
software that runs on the real G1, so the operator side is the same in sim and
on the robot.

```
 OPERATOR STATION                                     ROBOT (bringup g1.launch.py: sim now, real G1 later)
 keyboard_teleop /  ── /cmd_vel, 20 Hz ─────────────▶ walk_controller → policy → /lowcmd → motors
 gamepad_teleop
   status line      ◀── /g1/velocity, /g1/tilt ────── robot_status: measured motion and tilt
 camera window      ◀── /g1/head_camera/color/... ─── head camera (the D435i), compressed for Wi-Fi
 RViz               ◀── FAST-LIO, elevation map ───── what the robot's LiDAR has mapped
                    ── /g1/autonomy_mode ───────────▶ declared autonomy level: "teleop"
```

- **Commands:** each key press changes the speed one step, or the speed follows
  the stick. It is clamped to what the policy was trained on (the same ranges
  `walk_controller` clips to), and sent 20 times a second.
- **Status:** one line in the teleop terminal shows the command next to what the
  robot actually does, so the operator knows it is moving, stopped or fallen
  without seeing it.
- **Safety:**
  - The robot stands still until the first command. The controller starts locked.
  - `walk_controller` stops it 0.5 s after the last command, so a crashed or
    disconnected operator can't leave it walking.
  - Quitting teleop (Ctrl+C, closing the window) sends a stop.
- **One operator:** the status line warns if another program also sends `/cmd_vel`.
  The teleop scenarios make sure the sim itself sends none.

## [NEW] Set up once

You need Docker (and on Linux an NVIDIA GPU): see the root [README](../../../README.md).
Every `docker compose` command runs from your `humanoid` folder (e.g. `cd ~/humanoid`).
`<os>` is `linux` or `windows`. Inside the container the shell starts in the same
repo, at `/workspaces/humanoid`.

```bash
git switch sujith/teleop-main && git submodule update --init
docker compose -f docker/compose.<os>.yaml up -d --build      # builds the image the first time (~20 min)
docker compose -f docker/compose.<os>.yaml exec dev bash
cd ros2_ws && colcon build --symlink-install                  # inside the container; a few minutes
```

After a `git pull`, rebuild the same way. Python changes alone need no rebuild
(that is what `--symlink-install` is for). After a change to `docker/`, add `--build`
to `up -d` again.

## [NEW] Every session: three terminals

Start the container (`docker compose -f docker/compose.<os>.yaml up -d`). **Linux
only, once per login:** let the container open windows, and use `docker` without
`sudo` in a terminal opened before you joined the `docker` group:

```bash
xhost +SI:localuser:root
newgrp docker
```

Then three terminals, each a shell in the container
(`docker compose -f docker/compose.<os>.yaml exec dev bash`) with
`source ros2_ws/install/setup.bash`:

| Terminal | Command | What it is |
|---|---|---|
| 1. Robot | `ros2 launch bringup g1.launch.py scenario:=room_teleop` | The sim, FAST-LIO, the elevation map, RViz, `walk_controller`, `robot_status`. The MuJoCo window and RViz open |
| 2. Camera | `ros2 launch teleop operator.launch.py` | The operator's view: the head camera's compressed stream |
| 3. Operator | `ros2 run teleop keyboard_teleop` | Driving. **Click into this terminal before pressing keys** |

For the PS4 controller, terminal 3 is `ros2 run teleop gamepad_teleop`, plus a
fourth shell with `ros2 run joy game_controller_node`. See [PS4 controller](#ps4-controller).

What happens in terminal 1: the robot hangs on a gantry for a few seconds while
FAST-LIO starts, stands up, is let go, and then stands still and waits for you.
On a PC with two GPUs, pick the one for the sim with
`exec -e CUDA_VISIBLE_DEVICES=<n> dev bash` (`<n>` as `nvidia-smi` numbers them).

To practise like the competition, drive with only the camera window, RViz
(fixed frame `odom`) and the status line, never the MuJoCo window.

## [NEW] Tests

Each is a scenario in `sim/scenarios/` for terminal 1's `scenario:=`. All three
have no motion script (the sim publishes no `/cmd_vel`) and reset after a fall.

| Scenario | Where | Policy | Try |
|---|---|---|---|
| `room_teleop` | The LiDAR room: walls, boxes, pillars, four 20 cm steps | Flat walker (`g1_walk/model_12400.pt`) | Open floor; walk into a box to practise a reset |
| `course_teleop` | The competition course, start area, facing the door | Flat walker | The start area. The door's 7.7 cm sill and the step-over beams are beyond this policy |
| `course_teleop_heightmap` | The competition course, open deck, facing the crossing ramps | The team's terrain-aware walker (`best_g1_rough_heightmap/model_5996.pt`) | The ramps and the deck. Its terrain scan must not see the lane walls up close |

**What to check** in each:

1. **Driving:** forward, back, sideways and turning follow the keys or sticks,
   and `actual` on the status line roughly matches `cmd` (expect about 70–90%
   of the command).
2. **Stopping:** `space` (or letting go of the sticks) stops the robot, and
   `actual` settles near 0.
3. **Camera:** the camera window moves with the robot.
4. **Reset:** after a fall (`FALLEN` on the status line), SHARE or `r` stands it
   up where it fell. Driving works again after OPTIONS. The map in RViz stays put.
5. **Link lost:** `Ctrl+C` in terminal 1. The status line shows
   `ROBOT LINK LOST` within a second.
6. **One operator:** with both `keyboard_teleop` and `gamepad_teleop` running,
   each status line shows `WARNING: 1 other /cmd_vel source(s)`.

The scenario sets the policy. To try another model, edit the scenario or add
`policy_type:=` and `checkpoint:=` to terminal 1's command.

## Keys

| Key | Does |
|---|---|
| `w` / `s` | forward / back, 0.1 m/s per press (−0.5 to 1.0 m/s) |
| `a` / `d` | left / right, 0.1 m/s per press (±0.5 m/s) |
| `q` / `e` | turn left / right, 0.1 rad/s per press (±0.5 rad/s, the policy's trained range) |
| `space` | stop |
| **[NEW]** `r` | stop, and stand a fallen robot up (sim testing only) |
| `Ctrl+C` | stop the robot and quit |

## PS4 controller

`gamepad_teleop` drives the robot with a PS4 (DualShock 4) controller, laid out
like Unitree's own G1 remote: left stick walks, right stick turns. Use it
instead of `keyboard_teleop` (never both: the status line warns if two programs
send commands).

```bash
ros2 run joy game_controller_node    # reads the controller
ros2 run teleop gamepad_teleop       # drives the robot (its own terminal)
```

On Linux, plug it in (USB) or pair it (Bluetooth), then run both commands in
shells in the dev container: `compose.linux.yaml` passes `/dev/input` through,
including a controller plugged in after the container started. On Windows with
WSL, the controller first has to be passed into WSL and then into a container:
see [Controller on Windows (WSL)](#controller-on-windows-wsl).

| Control | Does | Unitree remote |
|---|---|---|
| left stick up / down | forward / back, up to 1.0 / 0.5 m/s | left stick |
| left stick left / right | step left / right, up to 0.5 m/s | left stick |
| right stick left / right | turn, up to 0.5 rad/s | right stick |
| OPTIONS | unlock: start driving | START |
| CIRCLE | stop and lock | L2 + B |
| **[NEW]** SHARE | stop, lock, and stand a fallen robot up (sim testing only) | |
| `Ctrl+C` (in its terminal) | stop the robot and quit | |

- **Starts locked:** the robot stands still until OPTIONS is pressed, and
  OPTIONS only works with both sticks let go, so unlocking never starts a move
  by itself. The status line ends with `LOCKED`, `DRIVING` or `NO CONTROLLER`.
- **Proportional:** speed follows how far the stick is pushed. Letting go stops
  the robot. A small dead zone ignores a stick resting slightly off centre.
- **Controller lost:** if no controller message arrives for 0.5 s (unplugged, out
  of range, driver closed), the robot stops, and stays locked when the
  controller comes back until OPTIONS is pressed again.
- **Other controllers:** `game_controller_node` gives every controller it
  recognises the same layout (an Xbox controller works too: Menu to start, B to
  stop, View to reset). Use it rather than `joy_node`, whose numbers change with
  the driver: on WSL, `joy_node` puts the right stick on axis 2 and the
  triggers, resting at +1, on axes 3 and 4. To change the numbers anyway:
  `ros2 run teleop gamepad_teleop --ros-args -p axis_turn:=2`
  (`axis_forward`, `axis_sideways`, `axis_turn`, `button_start`, `button_stop`, `button_reset`).

### Controller on Windows (WSL)

Tested with a wired DualShock 4. Bluetooth doesn't reach WSL, so use a USB cable.

**1. Pass the controller into WSL.** In PowerShell **as Administrator**, with
[usbipd-win](https://github.com/dorssel/usbipd-win) installed:

```powershell
usbipd list                          # find the controller's BUSID ("Wireless Controller", 054c:09cc)
usbipd bind --busid <BUSID>          # once per controller (add --force if it warns about a filter)
usbipd attach --wsl --busid <BUSID>  # again after every replug or WSL restart
```

While attached, Windows doesn't see the controller. `usbipd detach --busid <BUSID>`
gives it back.

**2. Load the joystick drivers in WSL** (after every WSL restart):

```bash
sudo modprobe usbhid joydev evdev
ls /dev/input/                       # js0 and event0 should be there
```

**3. Start the driver in its own container**, on the same network as the dev
container, from your `humanoid` folder (the dev container must be up):

```bash
docker run -d --rm --name joy --network humanoid_default \
  --device /dev/input/js0 --device /dev/input/event0 \
  humanoid-dev bash -c 'source /opt/ros/jazzy/setup.bash && ros2 run joy game_controller_node'
docker logs joy                      # "Opened game controller: PS4 Controller"
```

**4. Drive:** in a shell in the dev container, `ros2 run teleop gamepad_teleop`.
Stop the driver afterwards with `docker stop joy`.

## Status line

```
MODE TELEOP | cmd x +0.4 y +0.0 turn +0.0 | actual x +0.37 y -0.02 turn -0.07 | tilt  0.8 deg
```

| Part | Meaning |
|---|---|
| `MODE TELEOP` | the declared autonomy level |
| `cmd` | what you are commanding |
| `actual` | the robot's measured speed, in its own heading frame |
| `tilt` / `FALLEN` | degrees from upright; FALLEN above 45° |
| `ROBOT LINK LOST` | no status from the robot for 1 s |
| `WARNING: N other /cmd_vel source(s)` | another program is also commanding the robot |
| **[NEW]** `RESET: ...` | the sim's answer to SHARE / `r`, for a few seconds |

**[NEW]** `robot_status` (started by `bringup g1.launch.py`, on the robot side)
fills in `actual` and `tilt` from the robot's own topics, so it works the same
on the real G1:

- **tilt:** from the pelvis IMU's attitude in `/lowstate`.
- **turn:** the pelvis IMU's gyro, in `/lowstate`.
- **x, y:** FAST-LIO's position (TF `odom → pelvis`), averaged over 0.3 s and
  turned into the robot's heading frame. It lags a little, and is only as good
  as FAST-LIO's tracking.

It only publishes while `/lowstate` arrives, so `ROBOT LINK LOST` means the robot
itself has gone quiet. All values are per second of *sim* time, so they match
`cmd` even though the sim runs slower than real time.

## [NEW] Camera

The sim renders the head camera where the real D435i is (the URDF's `d435_link`:
on the head, 47.6° down), so mostly the ground ahead is in view. Its topics and
frames are the ones Intel's driver publishes on the robot
(`realsense2_camera`, with `camera_namespace:=g1 camera_name:=head_camera`):

| Topic | What |
|---|---|
| `/g1/head_camera/color/image_raw` | 640×360 RGB, about 29 frames per second of sim time |
| `/g1/head_camera/color/image_raw/compressed` | the same as JPEG, about a fifth of the bandwidth: what the camera window shows |
| `/g1/head_camera/color/camera_info` | the pinhole model, for drawing things into the image |

Frames: `d435_link → head_camera_link → head_camera_color_optical_frame`. To see
the raw stream instead: `ros2 launch teleop operator.launch.py image:=/g1/head_camera/color/image_raw`.

## [NEW] After a fall: reset

The flat walker can't climb steps or push past boxes. In the teleop scenarios a
fall doesn't end the session: press **SHARE** on the controller (or `r`). The
sim's gantry takes hold of the robot where it lies, lifts it upright over 3 s,
facing the same way, and it stands and walks again about 10 s after the press.
The lift is slow on purpose, so FAST-LIO keeps track and the map stays right.
Driving is locked until you press OPTIONS again.

This is for sim testing only. Only a scenario with `on_fall: respawn` offers it,
and `walk_controller` only accepts the reset when `g1.launch.py` runs such a
scenario. Other scenarios still end on a fall, and on the real robot the status
line shows `RESET unavailable`.

## [NEW] Things to know

- **The sim runs slower than real time:** about 0.4× in the room and 0.3× on the
  course, on an RTX 3070 / 16-core PC with all windows open. The robot moves in
  slow motion, and the camera shows about 9–12 frames per second.
  `viewer:=false` and `rviz:=false` on terminal 1's command close the windows.
- **Linux: the ROS network is private.** `compose.linux.yaml` sets
  `ROS_DOMAIN_ID=42` and keeps ROS discovery on this PC, so other robots or
  computers on your network can't send `/cmd_vel` to the sim, and teleop can't
  drive them. To reach a real G1, override both, e.g.
  `ROS_DOMAIN_ID_HUMANOID=0 ROS_DISCOVERY_RANGE=SUBNET docker compose -f docker/compose.linux.yaml up -d`.
- **Start the operator programs after terminal 1** if they show nothing: the
  map frame (`odom`) is only published once, at startup.

## Shut down

In this order, so the robot is always left standing still:

1. **Stop the robot:** `space` (or CIRCLE) in terminal 3. The status line shows
   `cmd x +0.0 y +0.0 turn +0.0`.
2. **Quit teleop:** `Ctrl+C` in terminal 3. It sends a final stop and prints `Stopped.`.
3. **Quit the camera and the stack:** `Ctrl+C` in terminals 2 and 1, and wait for
   the prompt to come back. The sim prints its report (FAST-LIO's and the
   elevation map's error against the truth) on the way out.
4. **Leave the container:** `exit` in each terminal.
5. **Stop the container**, from your `humanoid` folder:
   `docker compose -f docker/compose.<os>.yaml down`. The image and the build
   (in `ros2_ws/`, in your repo folder) stay, so next time `up -d` and
   `source ros2_ws/install/setup.bash` are enough.

If something hangs, a second `Ctrl+C` in its terminal usually ends it. If the
stack still won't stop, `docker compose -f docker/compose.<os>.yaml restart` from
your `humanoid` folder restarts the container and ends everything in it.

If teleop is closed any other way (the terminal window closed, or the program
killed), the robot still stops: teleop sends a stop on its way out, and
`walk_controller` stops the robot 0.5 s after the last command anyway.

## [NEW] On the real robot

The operator side doesn't change. On the robot side:

| | Sim | Real G1 |
|---|---|---|
| Walking, `/cmd_vel` | `walk_controller` | the same node |
| Status line | `robot_status`, from `/lowstate` and FAST-LIO | the same node |
| Camera | the sim renders it | `realsense2_camera` (to add to the image and to `g1.launch.py sim:=false`), plus a fixed transform `d435_link → head_camera_link` |
| Compressed video | `image_transport republish`, started by `g1.launch.py` | the driver's own |
| Reset after a fall | SHARE / `r` | none: people and the real gantry |
| Network | private to this PC | the robot's domain over Wi-Fi: see Things to know |

Expect the real camera to look different (exposure, blur, lighting) and to lag
by 100–300 ms over Wi-Fi.

## Topics

| Topic | Type | From |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | operator, 20 Hz |
| `/g1/autonomy_mode` | `std_msgs/String` | operator: `teleop`, latched |
| `/g1/velocity` | `geometry_msgs/TwistStamped` | robot (`robot_status`): measured velocity, heading frame |
| `/g1/tilt` | `std_msgs/Float32` | robot (`robot_status`): degrees from upright |
| `/g1/head_camera/color/*` | `sensor_msgs/Image`, `CompressedImage`, `CameraInfo` | robot: the head camera (see [Camera](#new-camera)) |
| `/sim/respawn` | `std_srvs/Trigger` | sim only, teleop scenarios: the reset |

## Files

| File | What |
|---|---|
| `teleop/keyboard_teleop.py` | the keyboard program: keys, `/cmd_vel`, status line, reset |
| `teleop/gamepad_teleop.py` | the PS4 controller program: sticks and buttons to `/cmd_vel`, same status line |
| `teleop/robot_status.py` | robot side: `/g1/tilt` and `/g1/velocity` from `/lowstate` and FAST-LIO |
| `launch/operator.launch.py` | the camera window |
| `test/` | unit tests: `python3 -m pytest test`, in this folder |
| `sim/scenarios/*_teleop*.yaml` (outside this package) | the scenarios for driving |
| `bridge/bridge/head_camera.py` (outside this package) | the sim's head camera |
| `bridge/bridge/motion.py`, `mujoco_g1_walk_sim.py` (outside this package) | the `operator` motion type, and the reset |
