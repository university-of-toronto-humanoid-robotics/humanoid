# Walk the G1 humanoid

Needs **Windows + an NVIDIA GPU + [Docker Desktop](https://www.docker.com/products/docker-desktop/)** (running).

## 1. Get the branch (once)

In your `humanoid` folder:

```bash
git pull
git switch mjlabtest1
git submodule update --init
```

The last line fills `unitree_rl_mjlab/` with Unitree's code (our repo only stores a link to it).

No `unitree_rl_mjlab` folder yet? `git clone --recurse-submodules -b mjlabtest1 https://github.com/University-of-Toronto-Humanoid-Robotics/humanoid.git`

## 2. Start the container

```bash
docker compose -f docker/compose.windows.yaml up -d --build
docker compose -f docker/compose.windows.yaml exec dev bash
```

The first build takes a while. Then, inside the container:

```bash
source /opt/mjlab/bin/activate
cd training/unitree_rl_mjlab
```

## 3. Walk the robot

```bash
python scripts/play.py Unitree-G1-Flat --viewer viser --checkpoint-file ../models/g1_walk/model_12400.pt
```

Open <http://localhost:8080> → **Commands → Twist → Enable** → move the sliders to steer.
`Ctrl+C` to stop.

Other models: change `12400` to `500`, `1000`, `2000` or `6000` to see it at earlier training stages.

## 4. Continue training from a model

```bash
mkdir -p logs/rsl_rl/g1_velocity/shared && cp ../models/g1_walk/model_12400.pt logs/rsl_rl/g1_velocity/shared/
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs 4096 --agent.logger tensorboard --agent.resume True --agent.load-run shared
```

## 5. Train your own from scratch

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs 4096 --agent.logger tensorboard
```

`Ctrl+C` stops training. A checkpoint is saved every 100 iterations in `logs/rsl_rl/g1_velocity/<date>/`.
Play one like in step 3, with `--checkpoint-file logs/rsl_rl/g1_velocity/<date>/model_<N>.pt`.

Tips:
- `<...>` means replace it (brackets included) with your folder or number. **Tab** autocompletes.
- Out of GPU memory? Use `--env.scene.num-envs 2048`.
- Keep training after closing the terminal: put `nohup` in front and `> train.log 2>&1 &` at the end.
