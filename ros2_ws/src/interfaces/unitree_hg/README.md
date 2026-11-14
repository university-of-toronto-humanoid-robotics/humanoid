# unitree_hg

The message definitions for the G1's low-level interface. They are the ones in Unitree's
[unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) (`cyclonedds_ws/src/unitree/unitree_hg/msg`, BSD
3-Clause, commit `668d1ec`), and only the five the low-level interface needs. Don't change them: they must match
what the robot sends, field for field.

The robot publishes `/lowstate` (joint angles and speeds, the pelvis IMU) and obeys `/lowcmd` (a target, gains and
torque for each motor). With CycloneDDS as ROS's middleware (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`) these are
ordinary ROS 2 topics. The walk sim publishes and reads the same two.
