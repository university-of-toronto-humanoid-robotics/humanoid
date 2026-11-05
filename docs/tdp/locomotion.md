# TDP: locomotion part (TL;DR for the team lead)

About 180 words in total, for a 2-page TDP. Paste-ready LaTeX is in [`locomotion.tex`](locomotion.tex), references in [`locomotion.bib`](locomotion.bib). Perception and arm IK are written by their owners.

- **Method:** one reinforcement-learning policy (PPO) that sends joint targets to all 29 G1 joints at 50 Hz.
- **Based on:** Unitree's open-source G1 training code ([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)) in the mjlab framework ([mjlab](https://github.com/mujocolab/mjlab)), trained in GPU simulation (MuJoCo Warp).
- **What we add:**
  - a terrain height map from the CAD-based course map;
  - a crouch command for low buttons;
  - balance while the IK moves the arms;
  - training terrain matching the measured course, including a digital twin of the testbed.
- **Software table:** MuJoCo + Warp, mjlab, unitree_rl_mjlab, RSL-RL (4 rows).
- **Status:** simulation only. A flat-ground walking policy works; the terrain-aware policy is in training.

| Where in the template | How much |
|---|---|
| Introduction | 1 sentence |
| System Description → Motion Planning and Whole-Body-Control | 1 paragraph (~110 words) |
| Competition Strategy | 1 sentence |
| Experiments | 1 sentence |
| Software List | 4 rows |
| Team members | 1 line |
| References | 3 entries |
