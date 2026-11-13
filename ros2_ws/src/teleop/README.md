# teleop

The operator station: drive the G1 from a keyboard, using only what the robot
sees and reports. (depending on what we want can move to controller or headset)

Competition rule: the operator sits at a separate station and may not look at
the robot during a trial. Everything they need comes through this interface.

## How it works

```
 OPERATOR STATION                               ROBOT (simulator now, real G1 later)
 keyboard_teleop ── /cmd_vel, 20 Hz ──────────▶ ROS bridge → walking policy → joints
   status line   ◀── /g1/velocity, /g1/tilt ─── measured motion and tilt
 camera view     ◀── /g1/head_camera/* ──────── head camera (where the D435i is)
                 ── /g1/autonomy_mode ────────▶ declared autonomy level: "teleop"
```

- **Commands:** each key press changes the speed one step. The speed is clamped
  to what the policy was trained on, and sent 20 times a second.
- **Status:** one line in the teleop terminal shows the command next to what the
  robot actually does, so the operator knows it is moving, stopped or fallen
  without seeing it.
- **Safety:**
  - The robot stands still until the first command.
  - It stops 0.5 s after the last one, so a crashed or disconnected operator
    can't leave it walking.
  - Quitting teleop (Ctrl+C, closing the window) sends a stop.
- **One operator:** the status line warns if another program also sends `/cmd_vel`.
- **Sim and real:** the real G1 must publish and accept the same topics, to avoid changing operator setup.

## Run it in simulation

Needs Docker Desktop running, with WSL integration on for your Ubuntu distro. (Note: will try to setup directly in Ubuntu when I have the chance)

**[NEW] Where to run the commands:** every `docker compose ...` command below
must be run from your `humanoid` folder (the repo you cloned, e.g. `cd ~/humanoid`),
because it uses the path `docker/compose.windows.yaml`. Once you're inside the
container (after `exec dev bash`), the shell starts in the same repo, at
`/workspaces/humanoid`, so `cd training` and `cd ros2_ws` work from there.

**First time on this computer**, in your `humanoid` folder:

```bash
git switch sujith/teleoperation && git submodule update --init
docker compose -f docker/compose.windows.yaml up -d --build
```

Otherwise just start the container:

```bash
docker compose -f docker/compose.windows.yaml up -d
```

**Terminal 1: the robot.** Open a shell in the container, then start the simulator:

```bash
docker compose -f docker/compose.windows.yaml exec dev bash
source /opt/mjlab/bin/activate && cd training
python scripts/play.py Unitree-G1-Flat --ros --viewer viser --num-envs 1 --checkpoint-file models/g1_walk/model_12400.pt
```

When the `viser (listening *:8080)` box appears, open <http://localhost:8080>.
The simulation only starts once a browser connects. Open **Camera Feeds** in the
side panel: that is the operator's view.

**Terminal 2: the operator.** In a second shell in the container:

```bash
docker compose -f docker/compose.windows.yaml exec dev bash
cd ros2_ws
colcon build --packages-select teleop
source install/setup.bash
ros2 run teleop keyboard_teleop
```

**Click into this terminal before pressing keys.** If the browser has focus,
W/A/S/D move the viewer's 3D camera instead of the robot.

To practice like the competition, drive using only Camera Feeds and the status
line, never the 3D view.

### Shut down

In this order, so the robot is always left standing still:

1. **Stop the robot:** press `space` in terminal 2. The status line shows
   `cmd x +0.0 y +0.0 turn +0.0`, and `actual` settles near 0.
2. **Quit teleop:** `Ctrl+C` in terminal 2. It sends a final stop and prints
   `Stopped.`.
3. **Quit the simulator:** `Ctrl+C` in terminal 1, and wait for the prompt to
   come back. Close the browser tab.
4. **Leave the container:** type `exit` in both terminals.
5. **Stop the container**, from your `humanoid` folder:

   ```bash
   docker compose -f docker/compose.windows.yaml down
   ```

   This keeps the image and the teleop build (it lives in `ros2_ws/`, in your
   repo folder), so next time `up -d`, then `source install/setup.bash` is enough.

If something hangs, a second `Ctrl+C` in its terminal usually ends it. If the
simulator still won't stop, run `docker compose -f docker/compose.windows.yaml restart`
from WSL, which restarts the container and ends everything in it.

If teleop is closed any other way (the terminal window closed, or the program
killed), the robot still stops: teleop sends a stop on its way out, and the
simulator stops the robot 0.5 s after the last command anyway.

### Model in Use

`Unitree-G1-Flat` with `models/g1_walk/model_12400.pt` is Unitree's flat-ground
walker. Base Policy to test the teleop system, will swap to our policy once ready.
The head camera is added to it by `--ros`. When a model trained on our tasks
(`Flat`, `Open`, ...) is shared, swap the task name and `--checkpoint-file`. The
operator side stays the same.

## Keys

| Key | Does |
|---|---|
| `w` / `s` | forward / back, 0.1 m/s per press (−0.5 to 1.0 m/s) |
| `a` / `d` | left / right, 0.1 m/s per press (±0.5 m/s) |
| `q` / `e` | turn left / right, 0.2 rad/s per press (±1.0 rad/s) |
| `space` | stop |
| `Ctrl+C` | stop the robot and quit |

## Status line

```
MODE TELEOP | cmd x +0.4 y +0.0 turn +0.0 | actual x +0.37 y -0.02 turn -0.07 | tilt  0.8 deg
```

| Part | Meaning |
|---|---|
| `MODE TELEOP` | the declared autonomy level |
| `cmd` | what you are commanding |
| `actual` | the robot's measured speed (body frame) |
| `tilt` / `FALLEN` | degrees from upright; FALLEN above 45° |
| `ROBOT LINK LOST` | no status from the robot for 1 s |
| `WARNING: N other /cmd_vel source(s)` | another program is also commanding the robot |

## Topics

| Topic | Type | From |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | operator, 20 Hz |
| `/g1/autonomy_mode` | `std_msgs/String` | operator: `teleop`, latched |
| `/g1/velocity` | `geometry_msgs/TwistStamped` | robot: measured velocity, body frame |
| `/g1/tilt` | `std_msgs/Float32` | robot: degrees from upright |
| `/g1/head_camera/image_raw` | `sensor_msgs/Image` | robot: head camera, `rgb8`, 640×360 |

## Files

| File | What |
|---|---|
| `teleop/keyboard_teleop.py` | the operator program: keys, `/cmd_vel`, status line |
| `launch/operator.launch.py` | the camera window |
| `test/test_keyboard_teleop.py` | unit tests: `python3 -m pytest test` |
| `training/g1_comp/camera.py` (outside this package) | the simulator's ROS bridge: `/cmd_vel` in; camera, velocity and tilt out |
