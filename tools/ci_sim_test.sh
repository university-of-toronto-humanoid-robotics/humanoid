#!/bin/bash
# CI's whole-stack test: builds the workspace, then walks the G1 a short way in the sim with no windows.
# Run it inside the dev container, from anywhere:   tools/ci_sim_test.sh
#
# It passes when the robot reaches the end of the ci_walk scenario without falling and FAST-LIO tracked it.
# It needs no GPU: elevation mapping fails to start without one, which is fine, because ci_walk's policy is
# blind. So this does not test the elevation map or /height_scan.
set -eo pipefail

MAX_DRIFT=0.20   # m, the most FAST-LIO's translation RMSE may be against the true pose
TIMEOUT=900      # s of real time before the launch is given up on

source /opt/ros/jazzy/setup.bash
cd "$(dirname "$0")/../ros2_ws"
colcon build --symlink-install --event-handlers console_cohesion+
source install/setup.bash

log=$(mktemp)
timeout "$TIMEOUT" ros2 launch bringup g1.launch.py scenario:=ci_walk viewer:=false rviz:=false 2>&1 | tee "$log" || true

fail() { echo "SIM TEST FAILED: $1"; exit 1; }
grep -q 'The robot fell' "$log" && fail 'the robot fell'
grep -q 'Completed 1 of 1 waypoints' "$log" || fail 'the robot did not reach the waypoint'
drift=$(grep -o 'translation RMSE [0-9.]*' "$log" | tail -1 | awk '{print $3}')
[ -n "$drift" ] || fail 'FAST-LIO published no odometry'
awk -v d="$drift" -v max="$MAX_DRIFT" 'BEGIN {exit !(d <= max)}' || fail "FAST-LIO drifted $drift m (limit $MAX_DRIFT m)"
echo "SIM TEST PASSED: reached the waypoint, FAST-LIO translation RMSE $drift m"
