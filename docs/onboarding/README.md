# Onboarding task

Shows that your setup works and that you can open a PR. Replace `<username>` with your GitHub username.

1. Set up and start the container ([README](../../README.md)).
2. On your machine, in the repo folder, create a branch:
   ```bash
   git checkout -b onboarding/<username>
   ```
3. Inside the container, create your file:
   ```bash
   python3 -c "import mujoco, rclpy, os; print('ROS', os.environ['ROS_DISTRO'], '| MuJoCo', mujoco.__version__)" > docs/onboarding/<username>.txt
   ```
4. On your machine, commit and push:
   ```bash
   git add docs/onboarding/<username>.txt
   git commit -m "Add <username> onboarding"
   git push -u origin onboarding/<username>
   ```
5. On GitHub, open a pull request into `main`. A lead will review and merge it.
