# Plan: HoST fall-recovery (standing-up) policy for the G1, in our MuJoCo/mjlab stack, trained on Killarney

## Summary
Port HoST ("Learning Humanoid Standing-up Control across Diverse Postures", Huang et al., RSS 2025, arXiv 2502.08378v2; code github.com/OpenRobotLab/HoST, mirror Sage-PhysicalAI/HoST) from Isaac Gym + its rsl_rl fork to our mjlab 1.2.0 (MuJoCo Warp) + rsl-rl-lib 5.0.1 stack, as a new task family `HoST-*` in a new package `training/g1_host/`. That means four critics (one per reward group), the L2C2 smoothness loss, the vertical pulling-force and action-bound curriculum, the 6-frame proprioceptive history, the paper's terrains (ground, platform, wall, slope), plus our own (course stairs, 15° ramps, prone/side/random-drop starts). We add a dead-simple Killarney (Slurm, H100) job-array workflow the PhD student can review and submit, and an evaluation script with the paper's metrics plus a "scenario gallery" (supine, prone, side, wall, ramp up/down, stairs, random drop, with videos). The end state is seeing it stand up reliably in all of them in sim.

## User Story
As the locomotion lead of the UofT G1 team,
I want a trained recovery policy that stands the G1 up from any fallen posture (flat, ramp, wall, stairs, supine, prone, side),
So that the operator can switch from the walking policy to the recovery policy after a fall instead of losing the run.

## Problem → Solution
The walking policies can't get up after a fall, nothing in `training/` simulates lying postures, our rsl_rl has a single critic, and the cluster scripts are untested generic-Alliance guesses (`def-CHANGEME`, repo in `$HOME`).
→ `HoST-*` tasks reproduce the paper's method in mjlab. A custom multi-critic PPO trains them. One `sbatch` array runs every task on its own H100 and auto-resumes. `eval_host.py` gives paper-style numbers and a gallery of videos for every scenario.

## Metadata
- **Complexity**: XL (about 25 new files, 3 small updates; new subsystem: env family + RL algorithm + cluster workflow). Suggest implementing in 3 PRs (see Notes).
- **Source PRD**: N/A (free-form request + paper PDF `C:\Users\markk\Downloads\2502.08378v2 (2).pdf`)
- **PRD Phase**: N/A
- **Estimated Files**: 22 new, 3 updated

---

## What the paper and code actually say (read this first)

Everything below was read from the paper (all 13 pages, incl. Tables II, VI and Appendix) **and** the released code. Where they disagree, **we follow the released code**: it is what produced the paper's numbers, and the HoST README thanks a reimplementer for finding "typos in the paper". Each choice is a named constant in `g1_host/params.py`, so switching to the paper's value is one edit.

### Method, as implemented
| Component | Value (code) | Where in HoST code |
|---|---|---|
| Observation (one frame, 76 for 23 DoF) | `[ω_base·0.25, projected_gravity(3), q(23, absolute), q̇(23)·0.05, a_{t-1}(23), β + U(−0.025,0.025)]` | `envs/base/host_ground.py:281-297` |
| History | 6 frames (current + 5), oldest first, 456 inputs | `g1_config_ground.py:65-69` |
| Unactuated start | first 30 policy steps (0.6 s; wall 50): actions ×0, obs ×0 → robot falls/settles damped. This is how lying postures are generated | `host_ground.py:95,296`, cfg `unactuated_timesteps` |
| Action law | `target = q_now + β·a`, τ = Kp·kpf·(target−q) − Kd·kdf·q̇; τ ← motor_strength·τ + offset, clipped to limits | `host_ground.py:424-456` |
| Actor | MLP 456→512→256→128→23, ELU, **Tanh output**, learnable std, init 0.8 | `rsl_rl/modules/actor_critic.py:58-69`, cfg `init_noise_std=0.8` |
| Critics | **4 separate** MLPs 456→512→256→1, ELU; critic input = actor input | `actor_critic.py:71-83` |
| Reward groups | `['task','regu','style','target']`, weights `[2.5, 0.1, 1, 1]` (prone: task 1) | `g1_config_ground.py:198-200` |
| Task group | **product** (starts at 1, ×head_height ×orientation), not dt-scaled | `host_ground.py:247-259` |
| Other groups | sum of terms × weight × dt (dt = 0.02) | `host_ground.py:261-268, 651-656` |
| Multi-critic advantages | GAE per critic (γ 0.99, λ 0.95); each critic's advantage normalized over the batch; A = Σ wᵢ·Âᵢ | `rsl_rl/storage/rollout_storage.py:129-146` |
| PPO | lr 1e-3 adaptive (desired_kl 0.01), clip 0.2, entropy 0.01, 5 epochs × 4 minibatches, 50 steps/env/iter, max_grad_norm 1, clipped value loss (mean over critics) | `legged_robot_config.py:186-205`, `ppo.py` |
| L2C2 | s̄ = s + u·(s_{t+1}−s)·not_done, u ~ U(−1,1); loss = cπ‖μ(s)−μ(s̄)‖² + cV‖V(s)−V(s̄)‖² (all 4 critics). Effective cπ = ub·lb/(ub−lb) = **0.111**, cV = 0.1·cπ = **0.0111** (wall: 0.25·cπ) | `rsl_rl/algorithms/ppo.py:182-195`, cfg `smoothness_upper_bound=1.0, lower_bound=0.1, value_smoothness_coef=0.1` |
| Pulling force | 100 N up on `torso_link` **and** 100 N on a virtual point 0.2 m above it (200 N total ≈ 60% of weight). Only after the unactuated phase and only if pelvis projected-gravity z < −0.8 (trunk near vertical). Prone: no orientation condition | `host_ground.py:106-115`, cfg `curriculum` |
| Curriculum | at each reset batch: if mean head height (head z − mean feet z) at episode end > 0.9 m → force −= 20 N (≥0), β −= 0.02 (≥0.25). Start 200 N, β = 1 | `host_ground.py:505-508` |
| Episode | 10 s (500 steps), `init_at_random_ep_len=True`, terminate only on time-out and blow-ups (max \|q̇\| > 300, \|v_base\| > 20) | `host_ground.py:168-179` |
| PD (sim) | Kp/Kd: hip 150/4, knee 200/6, ankle 40/2, shoulder/elbow/waist/wrist 100/4; sim dt 5 ms, decimation 4 → 50 Hz policy, 200 Hz PD | `g1_config_ground.py:73-95` |
| Stages | H₁ = 0.45 m, H₂ = 0.65 m on pelvis height (world z on ground; feet-relative on slope with 0.40/0.60) | cfg `target_base_height_phase1/3` |
| Iterations | ground 12 000 (GIF at 10 000), platform 12 000, wall 7 500 (GIF at 4 000), slope 10 000 (GIF at 8 000), prone 12 000 | `g1_config_*.py` `max_iterations`, README GIF names |
| Eval protocol | β fixed 0.25, no pull force, no obs noise, DR on, 5 s episodes; success = pelvis > 0.7 m (slope 0.6) ever **and** never back below 0.5 after; 5 × 250 episodes | `scripts/eval/eval_ground.py`, paper App. A |

### Paper ↔ code disagreements (we use the code value; constant names in parentheses)
| Item | Paper | Code | Constant |
|---|---|---|---|
| Joint velocity penalty | −1e−4 | −1e−3 | `W_REGU_DOF_VEL` |
| Joint tracking error | −2.5e−1 | −2.5e−4 | `W_REGU_TRACKING` |
| CoM offset Z | ±0.08 m | ±0.06 m (0.03×2) | `COM_OFFSET_Z` |
| Control delay | U(0,100) ms | 0–4 sim substeps = 0–20 ms | `DELAY_SUBSTEPS` |
| L2C2 λπ, λV | 1, 0.1 | effective 0.111, 0.0111 | `L2C2_POLICY_COEF`, `L2C2_VALUE_COEF` |
| L2C2 u | U(·) | U(−1,1) masked by not-done | in `algorithm.py` |
| Base height post-reward | exp(−20‖h−h*‖²) | exp(−20\|h−h*\|) | `post_base_height` |
| Platform height | 20–92 cm | 0.10 + 0.4·d m, d∈[0.3,0.9] → 22–46 cm | `PLATFORM_HEIGHT` |
| Ankle-parallel threshold | var < 0.05 | var(10·z) < 0.05 (i.e. real var < 5e−4 m²) | `ankle_parallel` |
| Initial joint scale | 0.9–1.1 | ground 0.9–1.1; platform/wall/slope 0.8–1.2 | per task |

### What the paper does NOT show (it shapes our plan)
- **One policy per terrain.** HoST trained ground, platform, wall and slope separately. "Joint training over all terrains" and "joint training of supine and prone" are unchecked TODOs in their README. The paper (Sec. VI-D, VIII) says supine + prone together "negatively impacted performance". So: reproduce per terrain first, then our mixed tasks. The deliverable is **two policies** (supine-family, prone-family), picked automatically from the IMU (body-frame gravity x < 0 → on its back). A single all-in-one policy is an experiment.
- **Stairs, ramps at 15°, random falls** are ours, not the paper's.
- **Hardware notes (for later):** on the real G1 they raised hip/knee Kp to 200/275 (×1.33–1.5) and found β = 0.3 smoother than 0.25. Full-mesh ankle collisions caused a sim-to-real gap (use points/capsules, as our model already does).

### Port decisions specific to our stack (all constants in `params.py`)
1. **29-DoF robot, 23 policy joints.** Our G1 is the 29-DoF model. The policy drives HoST's 23 joints (6 per leg, waist yaw, 5 per arm incl. wrist roll). The other 6 (waist roll/pitch, wrist pitch/yaw) are held at 0 by stiff PD, like the 23-DoF robot's locked joints. `POLICY_JOINTS = "host23"`; `"all29"` is the variant. Same robot model and IMU (pelvis) as the walking policy.
2. **MuJoCo position actuators, not torch torques.** mjlab's default integrator `implicitfast` treats only the actuator's own damping implicitly. Computing HoST's PD in torch makes the damping explicit, which goes unstable on low-inertia joints at Kd = 4 (shoulder/wrist-roll armature ≈ 0.0044 → Kd·dt/I ≈ 4.5 > 2). So every joint uses `BuiltinPositionActuatorCfg` with HoST's Kp/Kd. Motor strength is folded into per-env Kp/Kd scales (s·kpf, s·kdf). The actuation offset (±5% τ_lim) goes in through `qfrc_applied`.
3. **PD target held for the policy step** (`TARGET_UPDATE = "policy_step"`). This matches the paper's Eq. 1 and the real robot (Unitree's low-level PD holds q_des between 50 Hz commands). HoST's code recomputes `q_now + βa` every 5 ms substep, which makes it a constant torque Kp·βa rather than a servo. That may be part of why they needed higher Kp on hardware. `"substep"` reproduces the code exactly; run both as an ablation.
4. **Whole-body contact with friction.** G1 collision geoms default to `priority=1, condim=6` in the MJCF, and Unitree's `FULL_COLLISION` sets condim 1 on everything except the feet. With priority, the robot geom's condim wins, so **back, hands, knees and pelvis would touch the ground without friction** and the robot could never push off. We set condim 3 on all robot geoms and enable world + self contact for all of them (the opposite of the walking tasks' `LEGS_ONLY_WORLD_CONTACT`). We also add box/capsule primitives for the torso back plate, shoulder bar and pelvis, fitted to the visual mesh bounds: the stock torso is a single r = 9 cm capsule, so the robot would roll like a log.
5. **Restitution U(0,1)** has no MuJoCo equivalent. It is not ported (documented in `PORTING.md`).
6. **Heights.** The faithful tasks use HoST's exact definitions: pelvis world z, head = torso_link + (0,0,0.45) minus mean feet z, slope feet-relative. Stairs and mixed tasks use a 2-ray downward raycast (ground under pelvis and under head, robot geoms excluded), so lying across stairs can't fake "standing". `HEIGHT_REF = "host" | "ground"`.

---

## UX Design

### Before
```
┌────────────────────────────────────────────────────────────────┐
│ Robot falls in sim/real → no recovery skill.                   │
│ Cluster: cluster/train.sbatch, untested; def-CHANGEME account; │
│ repo + venv in $HOME (Killarney can't run jobs from $HOME).    │
│ PhD student has to edit scripts and guess modules/paths.       │
└────────────────────────────────────────────────────────────────┘
```

### After
```
┌─────────────────────── PhD student (Killarney) ────────────────────────┐
│ once:  cd $SCRATCH && git clone ... && bash humanoid/training/cluster/ │
│        killarney/setup.sh                                              │
│ test:  sbatch --account=aip-<PI> training/cluster/killarney/smoke.sbatch│
│        → "SMOKE PASS" + measured steps/s + hours per task               │
│ train: sbatch --account=aip-<PI> training/cluster/killarney/host.sbatch │
│        (array: one H100 per task; re-run the same line to resume)       │
│ share: bash training/cluster/killarney/collect.sh → one .tar.gz        │
└────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── You (laptop) ───────────────────────────────┐
│ python scripts/play.py HoST-Ground --viewer viser --checkpoint-file X  │
│ python scripts/eval_host.py --task HoST-Ground --checkpoint X          │
│   → Table-III-style numbers (success, feet, smoothness, energy)        │
│ python scripts/eval_host.py --gallery --supine A --prone B --video     │
│   → success per scenario + logs/videos/host/<scenario>.mp4             │
└────────────────────────────────────────────────────────────────────────┘
```

### Interaction Changes
| Touchpoint | Before | After | Notes |
|---|---|---|---|
| Task names | Flat, Open, … | + `HoST-Ground`, `HoST-Prone`, `HoST-Platform`, `HoST-Wall`, `HoST-Slope`, `HoST-Stairs`, `HoST-Supine-Mixed`, `HoST-Prone-Mixed`, `HoST-All` | Same `train.py` / `play.py` as today |
| Cluster | generic, untested | `cluster/killarney/` (setup, smoke, array job, collect) | Account passed on the sbatch command line, no file edits |
| Testing | `eval_policy.py` (walking) | `eval_host.py` (paper metrics, robustness sweeps, gallery, video) | |
| Checkpoint | `model_N.pt` + `policy.onnx` | + `policy_meta.json` (obs layout, joint order, gains, β) | Needed later for switching on the robot |

---

## Mandatory Reading

Line numbers are on the local `competition` branch (`origin/main` differs only in `__init__.py`, `train.py`, `play.py`; our code touches none of those).

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `training/g1_comp/actions.py` | 44-150 | Custom mjlab action term pattern (cfg dataclass + `build`, `process_actions`, `reset`) |
| P0 | `training/g1_comp/tasks.py` | 131-148, 151-278, 393-431 | Events/DR wiring, cfg mutation style, `runner_cfg`, `TASKS` dict |
| P0 | `training/g1_comp/__init__.py` | 59-67 | Registration loop with `register_mjlab_task` |
| P0 | `training/g1_comp/worlds.py` | 111-157, 247-311 | Building terrain from MjSpec boxes/meshes; `SubTerrainCfg.function` → `TerrainOutput` |
| P0 | mjlab `envs/manager_based_rl_env.py` | 339-419, 489-527 | Step order: `process_action` → per-substep `apply_action` → termination → reward → reset (curriculum first) → obs |
| P0 | mjlab `managers/reward_manager.py` | 118-133 | `_step_reward[:, i]` = value × weight (unscaled); our runner builds groups from it |
| P0 | mjlab `rl/runner.py`, `rl/vecenv_wrapper.py` | all | Runner interface train/play expect; wrapper returns `(TensorDict obs, rew, dones, extras)` with `extras["time_outs"]` |
| P0 | unitree `scripts/train.py` | 37-140 | `runner_cls(env, asdict(agent_cfg), log_dir, device)`, `.add_git_repo_to_log`, `.load`, `.learn(n, init_at_random_ep_len=True)` |
| P0 | unitree `scripts/play.py` | 179-185 | `runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)`, `get_inference_policy(device)` |
| P0 | unitree `src/assets/robots/unitree_g1/g1_constants.py` | all | Actuator classes, armature, effort/velocity limits, `FULL_COLLISION` |
| P1 | mjlab `actuator/builtin_actuator.py`, `envs/mdp/dr/actuator.py` | 32-92; 23-120 | Position actuators; how per-env gains are written (reuse for our Kp/Kd/strength DR) |
| P1 | mjlab `entity/entity.py` | 904-1022 | `set_joint_position_target`, `write_external_wrench_to_sim` (xfrc_applied, world frame, persists) |
| P1 | mjlab `managers/observation_manager.py` | 18-66, 320-360 | compute → noise → clip → scale → delay → history; single-term history = HoST layout |
| P1 | mjlab `terrains/terrain_generator.py` | 90-128, 246-277 | `curriculum=True` → one sub-terrain per column (cumsum of proportions), rows = difficulty |
| P1 | mjlab `envs/mdp/dr/body.py`, `dr/geom.py`, `envs/mdp/events.py` | body_mass, body_com_offset, pseudo_inertia; geom_friction; reset_root_state_uniform | Startup DR and resets |
| P1 | `training/g1_comp/sensors.py` | 30-66 | Raised raycast pattern (for the 2-ray height reference) |
| P1 | `training/cluster/train.sbatch`, `setup_env.sh` | all | Existing style; Killarney scripts mirror it |
| P2 | `training/scripts/check_task.py`, `record_video.py`, `profile_step.py` | all | Check/video/profiling script patterns |
| P2 | HoST reference (clone, see Task 1) | see "What the paper and code say" table | Formulas to port verbatim |

mjlab/rsl_rl source: inside the container at `/opt/mjlab/lib/python3.12/site-packages/{mjlab,rsl_rl}` (`docker exec humanoid-dev-1 ...`).

## External Documentation

| Topic | Source | Key Takeaway |
|---|---|---|
| HoST paper | arXiv 2502.08378v2 (local PDF) | Tables II, VI, Appendix A-B: all values above |
| HoST code | github.com/OpenRobotLab/HoST (= Sage-PhysicalAI/HoST) | Released configs differ from the paper in 10 places (table above) |
| Killarney access | [Mila PAICE docs](https://docs.mila.quebec/technical_reference/clusters/paice/) | `--account=aip-<PI>`; AIP allocations only (CIFAR AI Chairs / AI institutes) |
| Killarney hardware/login | [Alliance Killarney page](https://www.alliancecan.ca/en/services/compute/killarney) | 10 nodes × 8 H100 SXM 80 GB, 2× Xeon 6442Y, 2 TB RAM; 168 nodes × 4 L40S; login `killarney.alliancecan.ca` |
| Killarney usage | [Vector vec-playbook](https://github.com/VectorInstitute/vec-playbook) | `--gres=gpu:h100:1` works for one H100; partitions picked by `--time` (≤3 h, 12 h, 1 d, 3 d, 7 d); Duo MFA; **cannot submit or run from `$HOME`**, use `$SCRATCH`; scratch on request from ops-help@vectorinstitute.ai, purged after 60 days unused; home 50 GB; uv/venv recommended; compute nodes have internet (Mila table) |
| docs.alliancecan.ca/wiki/Killarney | blocked for automated fetch (bot check) | PhD student should skim it for the current driver/CUDA version and any QoS |

```
KEY_INSIGHT: torch==2.14.1 from PyPI is a CUDA 13.0 build (container shows 2.14.1+cu130) → needs NVIDIA driver ≥ 580.
APPLIES_TO: cluster/killarney/setup.sh
GOTCHA: Killarney's driver version isn't published where we could read it. smoke.sbatch prints it and fails loudly if torch.cuda.is_available() is False; setup.sh takes TORCH_INDEX_URL (e.g. https://download.pytorch.org/whl/cu128) as the fallback.

KEY_INSIGHT: rsl-rl-lib 5.0.1's PPO.construct_algorithm builds the critic with 1 output and RolloutStorage with rewards/values of width 1; advantages are normalized globally.
APPLIES_TO: algorithm.py / runner.py
GOTCHA: Subclassing would mean overriding construct_algorithm, storage, compute_returns and update anyway. We port HoST's small PPO (BSD-3/MIT) behind the runner interface instead.

KEY_INSIGHT: mjlab train.py/play.py construct a plain ManagerBasedRlEnv. There is no hook for an env subclass.
APPLIES_TO: everything env-side
GOTCHA: All HoST behaviour has to live in manager terms (action/obs/reward/event/curriculum/termination) and in our runner (reward groups).

KEY_INSIGHT: AiiDA is not needed. Plain sbatch is the whole workflow.
```

---

## Patterns to Mirror

### NAMING_CONVENTION
```python
# SOURCE: training/g1_comp/tasks.py:74-86 — module-level UPPER_CASE constants with a one-line why
STANDING_SHARE = 0.10  # robots told to stand still (Unitree: 0.05)
COMMAND_DELAY_STEPS = (0, 6)  # joint commands arrive 0-30 ms late (5 ms steps), per episode
```
2-space indentation, `from __future__ import annotations`, module docstring that explains the *why* in plain words, `snake_case_env_cfg(play=False)` factories, task ids registered from a `TASKS` dict.

### CUSTOM_ACTION_TERM
```python
# SOURCE: training/g1_comp/actions.py:44-46, 70-79, 102-104, 145-147
@dataclass(kw_only=True)
class ArmOverrideJointPositionActionCfg(JointPositionActionCfg):
  ...
  def build(self, env: ManagerBasedRlEnv) -> ArmOverrideJointPositionAction:
    return ArmOverrideJointPositionAction(self, env)

class ArmOverrideJointPositionAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._env = env
    ...
  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
```

### EVENT_AND_DR_WIRING
```python
# SOURCE: training/g1_comp/tasks.py:146-148, 275-277
cfg.events["command_delay"] = EventTermCfg(
  func=dr.sync_actuator_delays, mode="reset", params={"lag_range": COMMAND_DELAY_STEPS}
)
cfg.events["pd_gains"] = EventTermCfg(
  func=dr.pd_gains, mode="reset", params={"kp_range": PD_GAIN_SCALE, "kd_range": PD_GAIN_SCALE}
)
```

### TASK_REGISTRATION
```python
# SOURCE: training/g1_comp/__init__.py:59-67
for _task_id, _make in TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_make(),
    play_env_cfg=_make(play=True),
    rl_cfg=runner_cfg(_task_id),
    runner_cls=VelocityOnPolicyRunner,   # ours: HostRunner
  )
```

### RUNNER_CFG
```python
# SOURCE: training/g1_comp/tasks.py:393-402
def runner_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = task_id.lower().replace("-", "_")
  ...
```

### SUBTERRAIN_BUILDER
```python
# SOURCE: training/g1_comp/worlds.py:124-131, 263, 311
def box(self, kind, x0, x1, y0, y1, top, shade=1.0) -> None:
  geom = self.body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
    size=((x1 - x0) / 2, (y1 - y0) / 2, (top - _BOTTOM) / 2),
    pos=((x0 + x1) / 2, (y0 + y1) / 2, (top + _BOTTOM) / 2))
  self.geoms.append(TerrainGeometry(geom=geom, color=_color(kind, shade)))
...
def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
...
return TerrainOutput(origin=origin, geometries=tile.geoms, flat_patches={"spawn": spawns})
```

### RUNNER_LOAD (what our runner must support)
```python
# SOURCE: training/scripts/record_video.py:47-50
runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
obs = env.get_observations()          # TensorDict; policy(obs) -> actions
```

### CHECK_SCRIPT (our "test" pattern; no pytest in the container)
```python
# SOURCE: training/scripts/check_task.py:1-8, 27-37
"""Build a task and step it without a policy, to check that it works. ... From training/:
  python scripts/check_task.py G1-Comp-Course --num-envs 8 --steps 200
"""
import _setup  # noqa: F401
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
```

### LOGGING_PATTERN
```text
# SOURCE: rsl_rl on_policy_runner log lines, grepped by training/README.md:33
grep -o "Learning iteration *[0-9]*/[0-9]*" logs/flat_base.log | tail -1
```
Our runner prints the same `Learning iteration N/M` header and rsl_rl-style `key: value` lines, and writes TensorBoard scalars under `Loss/`, `Train/`, `Episode_Reward/`, `Curriculum/`, `Metrics/`.

### ERROR_HANDLING
```python
# SOURCE: training/scripts/train.py:52-54 — fail early with a plain-language SystemExit
if not source.is_file():
  raise SystemExit(f"--from: no such file: {source}")
```
Shell: `set -euo pipefail`, explicit checks with a one-line fix in the message (see `cluster/setup_env.sh:21-25`).

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `training/g1_host/__init__.py` | CREATE | Register `HoST-*` tasks with `HostRunner` |
| `training/g1_host/params.py` | CREATE | Every paper/code number in one place, each with its source line |
| `training/g1_host/robot.py` | CREATE | G1 29-DoF for HoST: position actuators with HoST gains, full contact with friction, extra torso/pelvis primitives, joint lists, keypoints |
| `training/g1_host/terrains.py` | CREATE | The "CADs": flat, platform, wall, slope/ramp, course stairs; per-task and mixed worlds |
| `training/g1_host/actions.py` | CREATE | `HostJointAction`: β rescale, held target, delay, unactuated phase, actuation offset, pulling force, wall push-back, held joints |
| `training/g1_host/observations.py` | CREATE | `host_proprio` frame (noise + unactuated mask inside), history 6 |
| `training/g1_host/heights.py` | CREATE | Base/head/feet heights, keypoints, "host" vs raycast "ground" reference |
| `training/g1_host/rewards.py` | CREATE | All HoST terms, `tolerance`, per-terrain weight lookup, group prefixes |
| `training/g1_host/events.py` | CREATE | Posture resets (supine, prone, side, sit, random drop), joint reset, per-reset gain/strength DR, startup DR wrappers |
| `training/g1_host/curriculum.py` | CREATE | Force/β curriculum + success metric logging |
| `training/g1_host/terminations.py` | CREATE | Blow-ups + per-terrain terminations |
| `training/g1_host/algorithm.py` | CREATE | Multi-critic actor-critic, storage (GAE per group), HostPPO with L2C2 |
| `training/g1_host/runner.py` | CREATE | `HostRunner`: rollout, group rewards, logging, save/load (incl. curriculum state), ONNX + meta export |
| `training/g1_host/tasks.py` | CREATE | Env cfg factories, `runner_cfg`, `TASKS` |
| `training/g1_host/PORTING.md` | CREATE | Checklist: every paper/code item → our file/constant, and every deviation with its reason |
| `training/scripts/_setup.py` | UPDATE | `import g1_host` (registers the tasks) |
| `training/scripts/check_host.py` | CREATE | Wiring checks (assertions) for every HoST task |
| `training/scripts/check_host_ppo.py` | CREATE | Algorithm math vs a transcribed HoST reference on random data |
| `training/scripts/bench_host.py` | CREATE | Steps/s and GPU memory at several env counts |
| `training/scripts/eval_host.py` | CREATE | Paper metrics, robustness sweeps, scenario gallery, videos |
| `training/cluster/killarney/README.md` | CREATE | One page for the PhD student |
| `training/cluster/killarney/setup.sh` | CREATE | One-time venv in `$SCRATCH` with pinned versions |
| `training/cluster/killarney/smoke.sbatch` | CREATE | 30-min check: GPU/driver, tasks build, 20 iterations, benchmark, PASS/FAIL |
| `training/cluster/killarney/host.sbatch` | CREATE | Job array, one H100 per task, auto-resume, final eval |
| `training/cluster/killarney/collect.sh` | CREATE | One tarball of results to send back |
| `training/cluster/README.md` | UPDATE | One line pointing Killarney users to `killarney/` |
| `training/README.md` | UPDATE | Section "Fall recovery (HoST)": tasks, commands, what to look at in TensorBoard |

## NOT Building

- Policy switching on the robot, the ROS side, the SLAM integration, sim-to-real deployment (out of scope per request). We only write `policy_meta.json` so that step has a contract.
- Hardware gain changes (Kp ×1.33–1.5), real-robot tests.
- Mirror augmentation (`g1_comp/symmetry.py`), RND, the paper's baselines (w/o-MuC, w/o-Force, Bound0.25, History ablations). The ablation switches exist as constants, but we don't build ablation tooling beyond the target-update A/B.
- UMAP / 3D-trajectory visualizations (Fig. 4).
- Restitution randomization (no MuJoCo equivalent).
- Multi-GPU training per job (mjlab's torchrunx workers don't see our tasks; one H100 per job is the design).
- AiiDA, Apptainer images (venv is enough; Apptainer only as a documented fallback).
- Any change to `g1_comp/`, `train.py`, `play.py`, `unitree_rl_mjlab/`, `docker/`, `ros2_ws/` or teammates' files.

---

## Step-by-Step Tasks

### Task 1: Branch and reference checkout
- **ACTION**: Start a feature branch; get the HoST code next to you for reading.
- **IMPLEMENT**: Branch `host-recovery` off `origin/main` (main is the superset of `competition`; the new package touches none of the files that differ). Your uncommitted `-Posture` work on `competition` is unaffected: commit or stash it first. `git clone --depth 1 https://github.com/OpenRobotLab/HoST.git` **outside** the repo (e.g. `%TEMP%\HoST`); read only, never import it.
- **GOTCHA**: Don't copy HoST files wholesale. Port the functions, with a header comment crediting HoST (MIT) and rsl_rl (BSD-3).
- **VALIDATE**: `git status` clean except new files; HoST tree readable.

### Task 2: `params.py`, the paper table as code
- **ACTION**: Create the single source of every number.
- **IMPLEMENT**: Sections with a source comment on each line (`# paper Tab.VI / host_ground.py:1107` style):
  - Control: `SIM_DT=0.005`, `DECIMATION=4`, `EPISODE_S=10.0`, `UNACTUATED_STEPS={"default":30,"wall":50}`, `KP={"hip":150,"knee":200,"ankle":40,"shoulder":100,"elbow":100,"waist":100,"wrist":100}`, `KD={"hip":4,"knee":6,"ankle":2,...:4}`, `HELD_KP/KD` (waist roll/pitch 200/5, wrist pitch/yaw 20/0.5; tune in check), `TARGET_UPDATE="policy_step"`, `POLICY_JOINTS="host23"`.
  - Observations: `OBS_SCALE_ANG_VEL=0.25`, `OBS_SCALE_DOF_VEL=0.05`, `HISTORY=6`, noise `ANG_VEL=0.2, GRAVITY=0.05, DOF_POS=0.01, DOF_VEL=1.5, BETA=0.025` (uniform ±).
  - Curriculum: `PULL_FORCE_START=200.0` (2×100 N), `PULL_FORCE_STEP=20.0`, `BETA_START=1.0`, `BETA_STEP=0.02`, `BETA_MIN=0.25`, `CURRICULUM_HEAD_HEIGHT=0.9`, `FORCE_UPRIGHT_GZ=-0.8`, `WALL_PUSHBACK_N=50.0`.
  - Stages and targets: `H_STAGE1=0.45`, `H_STAGE2=0.65`, `BASE_HEIGHT_TARGET=0.75` (slope 0.40/0.60/0.70), `HEAD_HEIGHT_TARGET=1.0`, `HEAD_MARGIN=1.0`, upper-body target pose (shoulder roll L +0.3 / R −0.3, all else 0), reset pose (hip pitch −0.1, knee 0.3, ankle pitch −0.2, elbow 0.8; platform knee 1.0; wall hip pitch −1.5, knee 0, elbow 0.7).
  - Rewards: every term weight from the table in Task 8, `GROUPS=("task","regu","style","post")`, `GROUP_WEIGHTS=(2.5,0.1,1.0,1.0)` (prone `(1.0,0.1,1.0,1.0)`).
  - DR: payload `(-2,5)` kg on torso_link (add), CoM XY `±0.12`, Z `±0.06`, link mass scale `(0.8,1.2)`, friction `(0.1,1.0)`, kp/kd scale `(0.85,1.15)` (prone `(0.85,1.25)`), motor strength `(0.9,1.1)`, actuation offset `±0.05·τ_lim`, `DELAY_SUBSTEPS=(0,4)`, initial joint scale `(0.9,1.1)` / `(0.8,1.2)`, offset `±0.1`.
  - PPO: `ACTOR=(512,256,128)`, `CRITIC=(512,256)`, `INIT_STD=0.8`, `LR=1e-3`, `DESIRED_KL=0.01`, `CLIP=0.2`, `ENTROPY=0.01`, `EPOCHS=5`, `MINIBATCHES=4`, `GAMMA=0.99`, `LAM=0.95`, `STEPS_PER_ENV=50`, `MAX_GRAD_NORM=1.0`, `L2C2_UPPER=1.0`, `L2C2_LOWER=0.1`, `L2C2_VALUE_COEF=0.1` (wall 0.25).
  - Iterations per task: Ground 12000, Prone 12000, Platform 12000, Wall 7500, Slope 10000, Stairs 12000, Supine-Mixed 20000, Prone-Mixed 20000, All 25000.
- **MIRROR**: NAMING_CONVENTION.
- **GOTCHA**: Keep paper-vs-code alternatives as comments next to each constant (table above).
- **VALIDATE**: `python -c "import g1_host.params"` in the container.

### Task 3: `robot.py`, the G1 for standing up
- **ACTION**: Build the `EntityCfg` from Unitree's spec without editing the submodule.
- **IMPLEMENT**:
  - `spec_fn`: call Unitree `g1_constants.get_spec()`, then add collision primitives. Read the torso_link and pelvis visual-mesh vertex AABB from the compiled spec and add `torso_back_collision` (box, 90% of AABB), `shoulder_bar_collision` (capsule across both shoulder joints), `pelvis_box_collision` (box). All in class/group 3, condim 3.
  - Collisions: `CollisionCfg(geom_names_expr=(".*_collision",), condim=3, priority={feet:1, others:1}, friction=…)`, contype/conaffinity both world and self (no legs-only mask).
  - Actuators: for each Unitree group, a `BuiltinPositionActuatorCfg` with **HoST Kp/Kd** (by name substring hip/knee/ankle/shoulder/elbow/waist/wrist), Unitree effort limits and armature (keep realistic armature). Held joints (`waist_roll`, `waist_pitch`, `.*_wrist_pitch`, `.*_wrist_yaw`) get `HELD_KP/KD`.
  - `init_state`: supine (pos z 0.5, quat (w,x,y,z) = normalize(1,0,−1,0)), joint pose from params.
  - Constants: `POLICY_JOINT_NAMES` (23, **fixed order** = HoST URDF order mapped to our names; record it in meta), `HELD_JOINT_NAMES` (6), `HEAD_OFFSET=(0,0,0.45)` and `PULL_POINT_OFFSET=(0,0,0.2)` in torso_link frame, ankle keypoint offsets in ankle_roll_link frame `[(0,0,0),(0.035,0.1,0),(0.035,-0.1,0),(0.2,0,0),(-0.15,0,0)]` (HoST URDF lines 1204-1300; used only for the ankle-parallel variance).
- **MIRROR**: `tasks.py:155-159` (`dataclasses.replace(FULL_COLLISION, …)`).
- **IMPORTS**: `from src.assets.robots.unitree_g1 import g1_constants as g1`, `from mjlab.actuator import BuiltinPositionActuatorCfg`, `from mjlab.entity import EntityCfg, EntityArticulationInfoCfg`, `from mjlab.utils.spec_config import CollisionCfg`.
- **GOTCHA**: The MJCF default class sets `priority="1" condim="6"` on every collision geom. Our cfg must override condim on **all** of them, or lying contacts are frictionless. Self-collision pairs between adjacent links are already excluded by the MJCF; verify no constant self-contact at the default pose.
- **VALIDATE**: In `check_host.py`: total mass printed (expect ≈ 33-35 kg; pull 200 N ≈ 60% of weight), 23 + 6 joints found, new geoms present, zero contacts at a standing default pose in the air.

### Task 4: `terrains.py`, the "CADs"
- **ACTION**: SubTerrainCfgs whose tile origin is the spawn point and whose feature has a fixed orientation relative to it (as HoST does), with difficulty ∈ [0,1] mapped to HoST's ranges.
- **IMPLEMENT** (box/mesh geoms via MjSpec, like `worlds._Tile`; tile 8 × 8 m; floor plane):
  - `HostFlatTerrainCfg`: plane.
  - `HostPlatformTerrainCfg`: box 4 m (x) × 6 m (y) centred, top `0.10 + 0.4·(0.3+0.6d)` m (22–46 cm); returned origin on the platform top; the robot spawns at origin + x∈[1.5,1.6], y∈[−2.5,2.5] → trunk on the platform, legs over the +x edge.
  - `HostWallTerrainCfg`: incline behind the robot (toward −x), angle `atan(slope)` with slope from HoST's list `[0.25,0.5,0.75,1,2,2.5,5,7.5,10]` (14°–84°), capped at 1.0 m tall; robot at origin x∈[0,0.05].
  - `HostSlopeTerrainCfg`: inclined plane rising toward −x, slope `[0.02…0.26]` (1.1°–14.6°); `ExtSlope` variant to 17° for the course's 15° ramps; robot at x∈[−2.5,−1.5]; spawn z = surface + 0.5.
  - `HostStairsTerrainCfg` (ours): course stairs (riser 0.203, tread 0.24, 5 risers, landing 1.22, width 0.77 between side panels; constants copied with a comment pointing to `sim/competition/README.md`, **not** imported from `g1_comp`). Riser `0.10→0.215` with difficulty; spawn on the flight with yaw ∈ {head up, head down, sideways}.
  - Worlds: one `TerrainGeneratorCfg` per faithful task (`curriculum=False`, random difficulty per tile, as HoST). One for mixed tasks with `curriculum=True` (one sub-terrain per column, rows = difficulty, `max_init_terrain_level=num_rows−1`, no level-update term), plus a helper `terrain_name_per_env(env)` that replicates mjlab's column→type rule (`index/num_cols + 0.001 < cumsum(proportions)`).
- **MIRROR**: SUBTERRAIN_BUILDER.
- **GOTCHA**: mjlab builds one shared terrain and each env is its own MuJoCo world, so robots overlapping in xy is fine. Keep terrain geoms in geom group 0 so the height rays (`include_geom_groups=(0,)`) never hit the robot (robot visual meshes are group 2, collision group 3).
- **VALIDATE**: `python scripts/view_world.py HoST-Wall` (existing script: pictures in `logs/worlds/`). Eyeball platform edge, wall angles, stairs.

### Task 5: `actions.py`, `HostJointAction`
- **ACTION**: One action term (23 dims) that reproduces HoST's control law on position actuators.
- **IMPLEMENT**:
  - State per env: `beta` (init 1.0), `pull_force` (init 200), `steps_since_reset`, Kp/Kd/strength scales (set by the reset event), `offset_torque`, delay lag (0–4 substeps), ring buffer of β·a (`DELAY_SUBSTEPS[1]+1` deep).
  - `process_actions(a)`: store raw `a` (no clip; tanh mean + Gaussian noise, as HoST clip=100). If `steps_since_reset < unactuated` → a = 0. If `TARGET_UPDATE=="policy_step"`: `target = q_now + β·a` once (delay applied to the β·a stream per substep).
  - `apply_actions()` (every 5 ms substep): substep mode recomputes `target = q_now + delayed(β·a)`. Set position targets for the 23 joints; held joints target 0. Unactuated phase: target = q_now (damped fall, like HoST ground; wall: also zero `qfrc` = limp). Write `offset_torque` to `qfrc_applied` of the 23 dofs. **Pulling force**: if actuated and pelvis projected-gravity z < −0.8 (or always for prone): world force (0,0,pull/2) at torso_link COM-relative points torso_link origin and origin + R·(0,0,0.2). Implement as one `write_external_wrench_to_sim` on torso_link with F = (0,0,pull) and τ = Σ (pᵢ − p_com) × Fᵢ. Wall: −50 N in x during the unactuated phase. Otherwise zero the wrench.
  - `reset(env_ids)`: zero buffers/counter, sample lag; keep `beta`/`pull_force` (the curriculum owns them).
  - Properties for others: `beta`, `pull_force`, `actuated_mask`, `joint_targets`, `last_actions`, `last_last_actions`.
- **MIRROR**: CUSTOM_ACTION_TERM (subclass `JointPositionAction` with `use_default_offset=False, scale=1`).
- **IMPORTS**: `torch`, `mjlab.envs.mdp.actions.JointPositionAction/JointPositionActionCfg`, `mjlab.utils.lab_api.math` (quat apply).
- **GOTCHA**: `xfrc_applied` persists until overwritten, so write zeros when the force is off and on reset. Derived quantities in the substep are one substep stale (mjlab docstring); that's fine. The `episode_length_buf` is randomized by `init_at_random_ep_len`, so **never** use it for the unactuated phase; use our own counter.
- **VALIDATE**: `check_host.py`: during the first 30 steps the targets equal q (|τ| ≈ Kd·|q̇|); the force is non-zero only when upright and actuated; delay lag distribution in range.

### Task 6: `observations.py`
- **ACTION**: One term `host_proprio` that returns the 76-dim frame (23 DoF) with HoST's scales; history 6.
- **IMPLEMENT**: Frame = `[ω_pelvis_b·0.25, g_pelvis_b, q_23, q̇_23·0.05, a_{t−1} (raw), β + U(±0.025)]`. Add uniform noise **inside** the function (noise in raw units × scale: ang vel 0.2·0.25, gravity 0.05, q 0.01, q̇ 1.5·0.05), then multiply by `actuated_mask`, so the unactuated frames are exactly zero (HoST order). `ObservationTermCfg(func=host_proprio, history_length=6, flatten_history_dim=True)`. Groups "actor" and "critic" both use it (HoST critic = actor obs). Play config: noise off.
- **GOTCHA**: Put noise in the function, not `ObservationTermCfg.noise` (mjlab applies cfg noise after compute, which would make masked frames non-zero). Confirm mjlab's circular buffer order is oldest → newest, and that on reset it is filled with the first frame (zeros here).
- **VALIDATE**: obs shape `(N, 456)`; during the unactuated phase all zeros; after, the last 76 entries equal the current frame.

### Task 7: `heights.py`
- **ACTION**: The quantities rewards, curriculum, terminations and eval share.
- **IMPLEMENT**: `head_pos(env)` = torso_link pose · HEAD_OFFSET; `feet_pos(env)` = ankle_roll_link origins; `ankle_keypoints_z(env)`; `base_height(env, ref)`: "host" → pelvis world z (slope: pelvis z − mean feet z); "ground" → pelvis z − ray hit under pelvis. `head_height(env, ref)`: "host" → head z − mean feet z; "ground" → head z − ray hit under the head. Raycast sensor cfg `host_ground_rays` (2 rays from 2 m above pelvis/head, `include_geom_groups=(0,)`, `ray_alignment="world"`), added only when `HEIGHT_REF=="ground"`.
- **VALIDATE**: Standing default pose on flat ground: host and ground refs agree within 5 cm.

### Task 8: `rewards.py`
- **ACTION**: Port every term; names carry the group prefix (`task_`, `regu_`, `style_`, `post_`) so the runner can group them.
- **IMPLEMENT** (HoST formulas):
  - `tolerance(x, lo, hi, margin, value_at_margin)` = 1 in bounds, else exp(−0.5·(d·√(−2 ln v))²), d = distance/margin (`g1_utils.py:249-266`).
  - Task (weight 1 each; runner multiplies): `task_head_height` = tol(h_head, [1, ∞), 1, 0.1); `task_orientation` = tol(−g_z, [0.99, ∞), 1, 0.05).
  - Regu: dof_acc Σ((q̇_{t−1}−q̇_t)/dt)² −2.5e−7; action_rate −0.01; smoothness Σ(a_t−2a_{t−1}+a_{t−2})² −0.01; torques Σ τ² −2.5e−6; joint_power Σ\|τ\|\|q̇\| −2.5e−5; dof_vel −1e−3; tracking Σ(target−q)² −2.5e−4; dof_pos_limits Σ out-of-(0.9·lower, 0.9·upper) −100 (HoST scales each bound by 0.9, not around the centre); dof_vel_limits Σ clip(\|q̇\|−0.9·v_lim, 0, 1) −1. Computed over the 23 policy joints; τ from `actuator_force`.
  - Style: waist_deviation 1(\|q_waist_yaw\|>1.4) −10; hip_yaw_deviation 1(max\|q\|>1.4 ∨ min\|q\|>0.9) −10; hip_roll_deviation same −10; shoulder_roll_deviation 1(q_L<−0.02 ∨ q_R>0.02) −2.5 (wall: 0/0; slope: 0.15/−0.15); left/right_foot_displacement exp(−2·max(‖p_base,xy−p_foot,xy‖², 0.3))·1(z_foot<0.3)·1(h_base>H₂) +2.5 each; knee_deviation 1(max\|q\|>2.85 ∨ min q<−0.06) **per terrain** (ground/prone −0.25; platform/slope/wall −10); shank_orientation tol(mean (knee−foot)_z/‖·‖, [0.8, ∞), 1, 0.1)·1(h_base>H₁) +10 (slope: `feet_ori`, feet-relative gating; prone: `thigh_ori` = min over legs of (thigh−knee)_z/‖·‖, no gating); ankle_parallel mean_LR var(10·z_keypoints) < 0.05 +20 (not prone); feet_distance 1(‖p_L−p_R‖>0.9) −10; feet_stumble 1(∃ foot \|F_xy\|>3\|F_z\|) **per terrain** (0 ground; −25 platform/wall/slope/stairs), from a foot-vs-terrain `ContactSensorCfg`; style_ang_vel_xy exp(−2‖ω_xy‖²)·1(h_base>H₁) +1 (prone +25); lower_body_deviation exp(−2 Σ var over L/R of [−hip_roll, hip_pitch, −hip_yaw, knee]) +1 (platform, wall); hip_pitch_deviation 1(max\|q\|>0.9 ∨ min\|q\|>0.8) −10 (prone only).
  - Post (all ·1(h_base>H₂)): ang_vel_xy exp(−2‖ω_xy‖²) +10; lin_vel_xy exp(−5‖v_xy‖²) +10; feet_height_var exp(−2·max(10\|z_L−z_R\|, 0.2)) +2.5; upper_pose exp(−0.1 Σ(q_upper−q*)²) +10 (arms + waist yaw); orientation exp(−5‖g_xy‖²) +10; base_height exp(−20\|h_base−0.75\|) +10; prone + lower_body_deviation +10.
  - Per-terrain weights in mixed worlds: the term takes `weights_by_terrain: dict[str, float]` and multiplies by a per-env tensor built once from `terrain_name_per_env`. The cfg weight is 1.
- **GOTCHA**: The reward manager skips terms with weight 0 and does `nan_to_num`. Don't give task terms a 0 weight. The ankle-parallel ×10 is inside the variance (×100 effective), as in the code.
- **VALIDATE**: `check_host.py` "standing" probe (robot written upright at the target pose, held by fixed base for one step): task ≈ 1, post terms ≈ their weights, style penalties 0. "Lying" probe: task < 0.1, post = 0.

### Task 9: `events.py`, `terminations.py`, `curriculum.py`
- **ACTION**: Resets, DR, ends of episodes, the curriculum.
- **IMPLEMENT**:
  - Reset root: `reset_posture(env, env_ids, posture, pose_range)` with posture ∈ {supine (pitch −90°), prone (pitch +90°), side_left/right (supine then roll ±90°, ±20° noise), sit (upright, wall), random (uniform random quaternion, z 0.6–1.0 m)}, position = env origin + per-task x/y offsets (Task 4), zero velocity. Mixed tasks: a per-env posture mix (e.g. supine 50%, side 30%, sit 20% on wall columns only).
  - Reset joints: default × U(scale) + U(±0.1), clipped to soft limits, q̇ = 0 (`host_ground.py:458-482`). Random posture: uniform within ±50% of each joint range.
  - Reset DR (per episode): kp scale U(0.85,1.15), kd scale U(0.85,1.15), strength U(0.9,1.1) → write gains s·kpf·Kp, s·kdf·Kd to the position actuators (copy the write path `dr.pd_gains` uses for builtin actuators); offset torque U(±0.05)·τ_lim; delay lag U{0..4}. Stored on the action term.
  - Startup DR: `dr.pseudo_inertia` (or `body_mass` scale) U(0.8,1.2) on all bodies except torso_link; then torso_link `body_mass` add U(−2,5); `body_com_offset` torso_link XY ±0.12, Z ±0.06; `geom_friction` robot geoms U(0.1,1.0) (robot geoms have priority, so their friction is used).
  - Terminations: `time_out` (time_out=True), `joint_vel_out` (max\|q̇\| > 300 after unactuated), `base_vel_out` (‖v‖ > 20); platform: pelvis z < 0.2, mean foot z > 0.35 after unactuated; wall: head z < 0.2, mean foot z > 0.35; slope: feet-relative pelvis < −0.1.
  - Curriculum term `host_force_and_bound(env, env_ids)`: called by mjlab at the start of `_reset_idx` with the episode's final state. If mean `head_height` over `env_ids` > 0.9 → `pull_force[env_ids] = max(pull_force−20, 0)`, `beta[env_ids] = max(beta−0.02, 0.25)`. Returns log dict `{"pull_force": mean, "beta": mean, "head_height_end": mean, "success_rate": frac(head>0.9 and pelvis>H₂)}`. Play/eval config: β = 0.25, force 0, term removed.
- **MIRROR**: EVENT_AND_DR_WIRING.
- **GOTCHA**: mjlab also calls curriculum compute on the very first reset (all envs, head height ≈ 0): no-op, fine. Event order matters: posture root reset before joint reset, gain DR after. `body_mass` warns about inertia; prefer `pseudo_inertia(alpha_range)` for link scaling.
- **VALIDATE**: `check_host.py` forces a fake "success" (writes a standing state, then triggers a reset) and asserts force −20 and β −0.02 for those envs only.

### Task 10: `tasks.py` + `__init__.py` + `_setup.py`
- **ACTION**: Assemble `ManagerBasedRlEnvCfg` per task; register.
- **IMPLEMENT**: `_base_cfg(play, terrain)` builds the scene (robot from Task 3, terrain from Task 4, contact sensor for feet, rays if needed), actions, observations, rewards (terrain-specific weights), events, terminations, curriculum. `sim`: timestep 0.005, `iterations`/`ls_iterations` as mjlab G1 (10/20), `nconmax=96`, `njmax=1200` (lying = many contacts; tuned by check). `episode_length_s=10`, `decimation=4`, `is_finite_horizon=False`. `play=True`: β 0.25, no force, no noise, no DR, `nconmax` raised, 10 s episodes (shows repeated get-ups). Tasks: `HoST-Ground`, `HoST-Prone`, `HoST-Platform`, `HoST-Wall`, `HoST-Slope` (faithful), `HoST-Stairs`, `HoST-Supine-Mixed` (flat, platform, wall, slope to 17° head up/down, stairs; supine/side/sit), `HoST-Prone-Mixed` (flat, slope, stairs; prone/side), `HoST-All` (everything + random drop; experimental). `runner_cfg(task_id)`: `RslRlOnPolicyRunnerCfg(num_steps_per_env=50, max_iterations=ITERATIONS[task], save_interval=100, experiment_name=task_id.lower().replace("-","_"), logger="tensorboard", clip_actions=None, algorithm/actor/critic fields from params)`. `_setup.py`: add `import g1_host  # noqa: E402,F401`.
- **MIRROR**: TASK_REGISTRATION, RUNNER_CFG.
- **GOTCHA**: Registering builds every cfg at import (both play and train); keep factories cheap (no terrain compile at import).
- **VALIDATE**: `python -c "import _setup; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if t.startswith('HoST')])"` lists 9 tasks.

### Task 11: `algorithm.py`, multi-critic PPO with L2C2
- **ACTION**: Port HoST's rsl_rl (`actor_critic.py`, `rollout_storage.py`, `ppo.py`) to plain torch.
- **IMPLEMENT**:
  - `MultiCriticActorCritic(num_obs, num_actions, num_critics, actor_dims, critic_dims, init_std)`: actor `Sequential(Linear, ELU, …, Linear, Tanh)`, `critics = ModuleList` of G MLPs; `std = Parameter(init_std·ones)`; `act`, `act_inference` (tanh mean), `evaluate` → `[B, G]`, `log_prob`, `entropy`.
  - `MultiCriticStorage(T, N, obs_dim, act_dim, G, group_weights)`: obs, actions, rewards/values/returns/advantages `[T,N,G]`, dones `[T,N,1]`, logp, mu, sigma. `compute_returns(last_values, γ, λ)`: GAE vectorised over G; per-critic normalisation over all (T,N); `A = Σ w_g·Â_g`. `mini_batch_generator`: like HoST, uses steps 0..T−2 with `next_obs = obs[1:]`, `cont = 1−done`.
  - `HostPPO.update()`: per minibatch: ratio, clipped surrogate, clipped value loss (mean over G), entropy; adaptive LR on KL (same rule as rsl_rl); L2C2: `u = cont·(rand−0.5)·2`, `s̄ = s + u·(s'−s)`, `Lπ = ‖μ(s)−μ(s̄)‖²` mean, `LV = ‖V(s)−V(s̄)‖²` mean, `loss += cπ·Lπ + cV·LV` with cπ = ub·lb/(ub−lb), cV = value_coef·cπ; clip grad 1.0. Return dict incl. `l2c2_policy`, `l2c2_value`, `action_smoothness` (‖μ(s)−μ(s')‖, logging only).
  - `process_env_step`: timeout bootstrapping `r += γ·V·time_out` per critic.
- **GOTCHA**: HoST normalises advantages inside the step loop (harmless). Do it once after the loop. Use `torch.inference_mode()` for rollout and plain grad mode in `update`.
- **VALIDATE**: `scripts/check_host_ppo.py` (Task 13).

### Task 12: `runner.py`, `HostRunner`
- **ACTION**: The class train.py/play.py/record scripts instantiate.
- **IMPLEMENT**:
  - `__init__(env, train_cfg: dict, log_dir=None, device="cpu")`: obs from `env.get_observations()["actor"]`; read the group-of-term map from `env.unwrapped.reward_manager.active_terms` (prefix → group; assert every term has a known prefix); build the model and `HostPPO` from `train_cfg` + params; TensorBoard `SummaryWriter` if `log_dir`.
  - `_group_rewards()`: `R = reward_manager._step_reward` `[N, terms]`; task = prod of task columns; regu/style/post = sum of their columns × `env.unwrapped.step_dt`; returns `[N, 4]` in order `GROUPS`. The env's own scalar reward is ignored.
  - `learn(num_learning_iterations, init_at_random_ep_len=False)`: as rsl_rl (randomise `episode_length_buf` if asked), rollout 50 steps, compute returns, update, log (`Learning iteration i/total`, `Computation: N steps/s (collection x s, learning y s)`, losses, mean group rewards, `extras["log"]` keys incl. `Episode_Reward/*` and `Curriculum/*`), save `model_{it}.pt` every `save_interval`, and a final one.
  - `save(path)`: model, optimizer, iter, `infos={"env_state": {"common_step_counter"}, "host_curriculum": {"pull_force": tensor, "beta": tensor}}`; export `policy.onnx` (actor mean, input `obs [1,456]`, output `actions [1,23]`, opset 18, `dynamo=False`) and `policy_meta.json` (joint order, Kp/Kd, held joints, β eval 0.25, history order oldest→newest, frame layout and scales, IMU = pelvis, action law, task, iteration).
  - `load(path, load_cfg=None, strict=True, map_location=None)`: load actor (+critics/optimizer/iter unless `load_cfg={"actor": True}`); restore curriculum tensors if shapes match, else fill with their mean. Return infos.
  - `get_inference_policy(device)`: returns `lambda obs: model.act_inference(obs["actor"])`. Also `add_git_repo_to_log(path)` (store the path, record `git rev-parse HEAD` into `params/`), `export_policy_to_onnx`.
- **MIRROR**: RUNNER_LOAD; mjlab `rl/runner.py` save/load shape.
- **GOTCHA**: train.py's `--agent.max-iterations` means "this many more" when resuming (rsl_rl semantics); keep it. `_step_reward` is private: wrap access in one function with a clear error if mjlab renames it.
- **VALIDATE**: 20-iteration local run (Task 16) writes `model_0.pt`, `model_19.pt`, `policy.onnx`, `policy_meta.json`; `play.py --checkpoint-file` loads it.

### Task 13: `check_host.py` and `check_host_ppo.py`
- **ACTION**: The tests (assertion scripts; exit code 1 on failure, "PASS"/"FAIL" per check).
- **IMPLEMENT** `check_host.py <task|all> --num-envs 64`: build; obs `(N,456)`, action dim 23; mass and pull-force share; settle test (after the unactuated phase: supine \|g_x\|>0.7 with g_x<0; prone g_x>0; side \|g_y\|>0.7; pelvis z < 0.35 on ground); friction test (horizontal 100 N on a lying robot: displacement with friction 1.0 ≪ with 0.1); force only when upright and actuated; curriculum step on fake success; reward probes (standing/lying); contact/constraint use peak < 80% of `nconmax`/`njmax` over 500 random-action steps; no NaN; waist roll/pitch deflection < 0.1 rad under random actions; per-env terrain names on mixed tasks match spawn geometry; print a summary table.
  `check_host_ppo.py` (CPU ok): (1) vectorised GAE + per-group normalisation + weighting equals a loop transcription of HoST `rollout_storage.py:129-146` on random data (atol 1e−5); (2) L2C2 coefficients 0.1111/0.01111 (wall 0.02778); (3) L2C2 loss = 0 when next_obs = obs; (4) `tolerance` equals HoST's at 7 test points; (5) actor outputs ∈ [−1,1]; (6) ONNX export equals torch (atol 1e−5) if onnxruntime is available, else skipped with a note.
- **MIRROR**: CHECK_SCRIPT.
- **VALIDATE**: both print PASS for every task.

### Task 14: `eval_host.py`
- **ACTION**: How you test, the same way as other policies (play/video) plus numbers.
- **IMPLEMENT**:
  - `--task T --checkpoint X [--episodes 250 --repeats 5]`: paper protocol (β 0.25, force 0, noise off, DR on, `--seconds 5`). Metrics: E_succ (pelvis > 0.7 m; slope 0.6 m; ever reached and never < 0.5 after), E_feet (feet travel after reaching), E_smth (Σ‖q_{t+2}−2q_{t+1}+q_t‖², in degrees, per step), E_engy (Σ\|τ\|·\|q̇\|·dt until pelvis > H₂), time-to-stand. Print mean ± std over repeats as a Table-III-like row; write `eval/<task>_<iter>.json`.
  - `--robustness`: sweeps as Fig. 5 / Table V (CoM x offset −0.1…0.1 m, constant sagittal force 0…100 N, initial joint offset 0…0.3 rad, random torque dropout 0…0.2 at 200 Hz, payload 4…12 kg).
  - `--gallery --supine A --prone B`: fixed scenarios (flat supine/prone/left/right/random drop; platform 0.3 m; wall 30°/60°; slope 10° head-up/head-down; ramp 15° head-up/head-down/sideways; stairs head-up/head-down/sideways; random drop near stairs). Picks the policy per robot from body-frame gravity x at the end of the settle phase (supine < 0 → A, else B). Prints a table of success % per scenario (64 robots each, DR on); `--video` writes `logs/videos/host/<scenario>.mp4` (1 robot, offscreen render, like `record_video.py`).
- **MIRROR**: `record_video.py` (render, `imageio`), `eval_policy.py` (argparse + table).
- **VALIDATE**: Runs with zero actions (`--checkpoint` omitted): success 0%, videos show the robot settling.

### Task 15: `bench_host.py`
- **ACTION**: Answer "how many robots on an H100".
- **IMPLEMENT**: For `--num-envs 4096 8192 16384` (each in a fresh process): build the task, run 3 warm-up + 10 rollout iterations of 50 steps with random actions, plus 1 PPO update; report steps/s, s/iter, peak `torch.cuda.max_memory_allocated`, MuJoCo contact use; print the recommended N (largest with ≥1.3× the steps/s of the previous and < 70 GB) and hours per task = ITERATIONS × s/iter.
- **MIRROR**: `profile_step.py` timing helpers.
- **VALIDATE**: Laptop, 1024/2048 envs: finishes and prints numbers.

### Task 16: Local end-to-end smoke (laptop, container)
- **ACTION**: Prove the loop before sending anything to the cluster.
- **IMPLEMENT**: `check_host.py all`; `check_host_ppo.py`; `python scripts/train.py HoST-Ground --env.scene.num-envs 256 --agent.max-iterations 20`; then a 1024-env, 300-iteration run in the background. Expected in TensorBoard: `Curriculum/pull_force` starts at 200 and the head height at episode end rises. Robots sitting up and being lifted by the force within a few hundred iterations is the sign the rewards are wired. `play.py HoST-Ground --viewer viser --checkpoint-file ...`.
- **VALIDATE**: Files written, curves move, play shows motion; note laptop s/iter for the README.

### Task 17: Killarney scripts (`training/cluster/killarney/`)
- **ACTION**: The PhD-facing workflow, nothing to edit except passing the account.
- **IMPLEMENT**:
  - `setup.sh` (login node, once): refuse to run if the repo is under `$HOME` (Killarney can't run jobs from home); `module load python/3.12` (override `PYTHON_MODULE`); `python -m venv $SCRATCH/venvs/mjlab`; `pip install mjlab==1.2.0 mujoco-warp==3.5.0 mujoco==3.5.0 warp-lang==1.12.0 scipy==1.15.3 torch==2.14.1 rsl-rl-lib==5.0.1 tensordict==0.15.6 onnx==1.23.2 tensorboard imageio[ffmpeg]` (`TORCH_INDEX_URL` optional for a CUDA 12.x build); `python -c "import _setup; ..."` lists the HoST tasks; print "SETUP OK".
  - `smoke.sbatch` (`--gres=gpu:h100:1 --cpus-per-task=6 --mem=64G --time=0:30:00`): `nvidia-smi` (driver, GPU); torch/warp CUDA check (fail with "rerun setup with TORCH_INDEX_URL=…cu128" if not); `check_host.py all --num-envs 64`; `check_host_ppo.py`; `train.py HoST-Ground` 4096 envs × 20 iterations; `bench_host.py`; last line `SMOKE PASS` or `SMOKE FAIL: <step>`.
  - `host.sbatch` (`--array=0-4` default = Ground, Prone, Platform, Wall, Slope; `TASKS="..."` to change; `--gres=gpu:h100:1 --cpus-per-task=6 --mem=64G --time=1-00:00:00`, output `host-%A_%a.out`): `cd` to the repo under `$SCRATCH` (error if under `$HOME`); env vars `WARP_CACHE_PATH=$SCRATCH/.cache/warp`, `MUJOCO_GL=egl`, `WANDB_MODE=offline`; `TAG=${RUN_TAG:-host}_${TASK}`; if a run matching `.*_${TAG}` has a checkpoint → `--agent.resume True --agent.load-run ".*_${TAG}$" --agent.max-iterations $((TARGET-last))`, else fresh; `NUM_ENVS` (default 4096, or what smoke recommended); after training, `eval_host.py --task $TASK --checkpoint <last>` → json. Resubmitting the same `sbatch` line continues every unfinished task.
  - `collect.sh`: per run: `params/`, `events.out.tfevents*`, the last 2 `model_*.pt`, `policy.onnx`, `policy_meta.json`, `eval/*.json` → `$SCRATCH/host_results_<date>.tar.gz`; prints its size and an `scp` line.
  - `README.md`: prerequisites (CCDB account on the PI's `aip-` allocation, Duo, SSH key, scratch requested from ops-help@vectorinstitute.ai); 5 commands (clone into `$SCRATCH`, setup, smoke, host, collect); how to check (`squeue --me`, `tail -f host-<job>_<i>.out`, `grep "Learning iteration"`); what "done" looks like (`Curriculum/pull_force` reaches 0 and `beta` reaches 0.25, success_rate > 0.95); scratch is purged after 60 days, so move results; what each script does in two lines each, so the student can review quickly.
- **MIRROR**: `cluster/train.sbatch` (variables block, `set -euo pipefail`, ARGS array).
- **GOTCHA**: Pass `--account` on the command line (or `export SBATCH_ACCOUNT=aip-<PI>`); no `#SBATCH --account` line to forget. Warp compiles kernels on the first run (minutes); the cache in `$SCRATCH` makes later jobs fast. Don't rely on compute-node internet.
- **VALIDATE**: `bash -n` on every script; `shellcheck` if available; dry run of the resume logic with a fake run folder on the laptop (`TARGET` math, regex).

### Task 18: Docs
- **ACTION**: `training/README.md` section "Fall recovery (HoST)": the tasks table, the 3 commands (train, play, eval), what each TensorBoard curve means, and the two-policy plan. `cluster/README.md`: one line "On Killarney use `killarney/`". `g1_host/PORTING.md`: every row of "What the paper and code actually say" → file/function/constant, plus each deviation and why.
- **VALIDATE**: Commands in the docs run as written.

### Task 19: Training campaign (operational; after the PRs)
- **ACTION**: Phases, each with a fallback.
- **IMPLEMENT**:
  - **A (validate the port)**: smoke → `HoST-Ground` alone (12 000 iterations). Target: eval success ≥ 95% (paper 99.5%), smoothness of the same order as Table III. If it fails, debug before spending more GPU.
  - **B (paper reproduction, one array)**: Prone, Platform, Wall, Slope (+ Ground with `TARGET_UPDATE="substep"` as the A/B).
  - **C (the deliverable)**: `HoST-Supine-Mixed`, `HoST-Prone-Mixed` (20 000 each; fine-tune from the B policies with `--from` if that helps) and `HoST-All` (experiment). Pick the env count from the smoke benchmark (expect 8192 if it gives ≥1.3× throughput).
- **VALIDATE**: Acceptance criteria below, with the gallery.

---

## Testing Strategy

### Unit / wiring checks
| Test | Input | Expected Output | Edge Case? |
|---|---|---|---|
| GAE + multi-critic advantages | random T=50,N=8,G=4 with random dones/timeouts | equals HoST loop reference | yes (dones at t=0, T−1) |
| L2C2 coefficients | ub 1, lb 0.1, value 0.1/0.25 | 0.1111 / 0.01111 / 0.02778 | |
| L2C2 with s′ = s | identical batches | loss 0 | yes |
| tolerance | x in/out of bounds, margin 0 | HoST values | yes (margin 0) |
| Obs layout | any task | (N, 456); zeros while unactuated | yes (right after reset) |
| Settle postures | supine/prone/side/sit/random | IMU gravity signs as expected | |
| Friction on body contacts | lying robot pushed 100 N | slides far less at μ=1.0 than μ=0.1 | |
| Pull force gating | trunk tilted vs upright | 0 N vs pull_force | yes (exactly −0.8) |
| Curriculum | fake success subset | only those envs: −20 N, −0.02 β, floors held | yes (at floors) |
| Reward probes | standing vs lying | task≈1 & post≈weights vs task<0.1 & post=0 | |
| Contact buffers | 500 random steps, all tasks | < 80% nconmax/njmax | yes (stairs, random drop) |
| Resume | save at 20, load, train 10 | iteration 30, curriculum tensors restored | yes (different num_envs) |
| ONNX | export | matches torch | |

### Edge Cases Checklist
- [ ] Episode reset right when the curriculum floors are reached (force 0, β 0.25)
- [ ] `init_at_random_ep_len` (first episodes truncated) doesn't break the unactuated counter
- [ ] Time-out bootstrapping per critic
- [ ] Mixed world: every column has robots (num_envs ≥ num_cols)
- [ ] Play config (β 0.25, no force) loads a training checkpoint with `load_cfg={"actor": True}`
- [ ] Resume with a different `NUM_ENVS` (curriculum filled with the mean)
- [ ] Cluster: repo under `$HOME` → clear error; torch without CUDA → clear error

---

## Validation Commands

All from `training/` inside the container (`docker compose -f docker/compose.windows.yaml exec dev bash`, `source /opt/mjlab/bin/activate && cd training`).

### Static Analysis
```bash
python -m compileall -q g1_host scripts
bash -n cluster/killarney/*.sh cluster/killarney/*.sbatch
```
EXPECT: no output (no repo-wide type checker or linter is configured)

### Unit Tests
```bash
python scripts/check_host_ppo.py
python scripts/check_host.py all --num-envs 64
```
EXPECT: every line PASS, exit 0

### Full Test Suite (no regressions in the existing tasks)
```bash
python scripts/check_task.py Flat --num-envs 8 --steps 50
python scripts/check_task.py G1-Comp-Course --num-envs 8 --steps 50
python -c "import _setup; from mjlab.tasks.registry import list_tasks; print(len(list_tasks()))"
```
EXPECT: unchanged behaviour; task count = old count + 9

### Training smoke
```bash
python scripts/train.py HoST-Ground --env.scene.num-envs 256 --agent.max-iterations 20 --agent.run-name smoke
python scripts/bench_host.py HoST-Ground --num-envs 1024 2048
```
EXPECT: `model_19.pt`, `policy.onnx`, `policy_meta.json`; steps/s printed

### Manual Validation
- [ ] `python scripts/play.py HoST-Ground --agent zero --viewer viser`: robots drop and settle on their backs (each task: right posture/place)
- [ ] `python scripts/view_world.py HoST-Supine-Mixed`: platform edge, wall angles, slopes, stairs look right
- [ ] After the 300-iteration laptop run: robots sit up and get lifted; `pull_force` curve starts dropping later
- [ ] On Killarney: `smoke.sbatch` ends with `SMOKE PASS`
- [ ] After phase A: `eval_host.py --task HoST-Ground` success ≥ 95%; `play.py` shows a smooth get-up
- [ ] After phase C: `eval_host.py --gallery --video` ≥ 95% on every scenario; videos look smooth (no flailing or bouncing)

---

## Acceptance Criteria
- [ ] All 19 tasks done; all validation commands pass
- [ ] `HoST-Ground` reproduces the paper in sim: success ≥ 95%, β = 0.25 and force 0 reached during training
- [ ] Supine + prone policies with IMU selection: ≥ 95% success on every gallery scenario (DR on), videos for each
- [ ] PhD student runs setup → smoke → host → collect from the README alone, editing no file
- [ ] Existing tasks and scripts untouched in behaviour

## Completion Checklist
- [ ] Code follows `g1_comp` style (2-space, docstrings with the why, constants at the top)
- [ ] Every paper/code number lives in `params.py` with its source; `PORTING.md` complete
- [ ] Errors are plain-language `SystemExit`s / shell messages with the fix
- [ ] Logs keep the `Learning iteration N/M` format
- [ ] Check scripts cover every task
- [ ] No hardcoded paths outside `params.py` / script variables
- [ ] Docs updated (training/README.md, cluster/README.md, PORTING.md)
- [ ] Nothing in teammates' areas or the Unitree submodule changed

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Supine + prone (and all terrains) in one policy interfere (paper says so) | High | High | Deliverable = 2 policies + IMU selection; per-terrain tasks first; `HoST-All` is only an experiment |
| Coarse/frictionless collision model makes get-up impossible or unrealistic | High if missed | High | condim 3 everywhere, torso/pelvis primitives, friction test in `check_host.py` |
| Torch-side PD unstable on small joints | High if done naively | High | Built-in position actuators (implicit damping) |
| Killarney driver < 580 for torch cu130 | Medium | Medium | Smoke job detects it; `TORCH_INDEX_URL` cu12x fallback |
| Throughput lower than hoped (whole-body contacts) | Medium | Medium | Benchmark first; tune nconmax/njmax; 1-day jobs auto-resume; 3-day partition exists |
| Target-update semantics change learned behaviour vs HoST | Medium | Medium | Constant + A/B run in phase B |
| Curriculum fooled on stairs (head above feet while lying) | Medium | Medium | Raycast "ground" heights in stairs/mixed tasks |
| Held waist roll/pitch sag under load (29-DoF) | Medium | Low | Stiffer held gains; deflection check; `"all29"` variant |
| `_step_reward` (private) changes in a future mjlab | Low | Medium | Pinned mjlab 1.2.0; one accessor with a clear error |
| Scratch purge (60 days) loses checkpoints | Medium | High | `collect.sh` after each phase; copy to project/home |
| Curriculum state lost when a 24 h job resumes | High if missed | High | Saved in checkpoint `infos` and restored |

## Notes
- **Throughput estimate (to be replaced by the benchmark):** laptop Flat = 4096 × 24 steps in 6.7 s ≈ 14.6k steps/s. HoST needs 4096 × 50 = 204 800 steps per iteration. On the laptop that's ≈ 15–25 s/iter (≈ 2–3 days for 10 000 iterations, so the laptop is for smoke tests only). An H100 SXM has about 13× the memory bandwidth; expect roughly 2–4 s/iter, so **≈ 6–12 h per 10 000-iteration task**, inside the 1-day partition. All five phase-B tasks run at the same time on five H100s (array job), so phase B is about one day of wall time.
- **More envs on an H100:** HoST's curriculum steps per episode, not per iteration, so it is unaffected by env count. More envs mean bigger batches per update (4096 × 50 = 204 800 samples, 4 minibatches). Keep 4096 for phase A (faithful). For phase C, use 8192 if the benchmark shows ≥ 1.3× steps/s (likely on 80 GB), and expect ~0.6–0.7× the iterations. Multi-GPU per job: no (see NOT Building). Instead, one GPU per task, in parallel.
- **Suggested PR split:** (1) Tasks 1–13 + 16 for `HoST-Ground` only (env, algorithm, runner, checks); (2) the other terrains, mixed tasks, eval/gallery (Tasks 4 rest, 14); (3) Killarney scripts + docs (15, 17, 18). (3) is independent and can be written in parallel with (1).
- The real-robot integration (switching from the walking policy, ROS, gains) is the next prompt; `policy_meta.json` is its contract.
- Killarney facts came from the Alliance Killarney page, Mila's PAICE docs and Vector's vec-playbook. The official wiki page blocks automated reading, so the student should skim it once for anything newer (driver version, QoS).
