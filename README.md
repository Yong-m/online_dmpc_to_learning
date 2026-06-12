# Online DMPC to Learning

Multi-drone navigation via GPU-batched Distributed MPC (DMPC) and reinforcement learning.
A PolicyFlow actor (conditional rectified flow) is trained with DMPC velocity-alignment reward in Isaac Lab,
enabling decentralized collision-aware navigation without the DMPC solver at inference.

Code: https://github.com/Yong-m/online_dmpc_to_learning

---

## Repository Structure

```
online_dmpc_to_learning/
├── cpp/                          # C++ reference DMPC implementation (Luis et al. 2020)
├── matlab/                       # Visualization and benchmarking scripts
├── quadcopter/                   # Isaac Lab environments and GPU DMPC expert
│   ├── multi_drone_dmpc_env.py       # Base multi-drone environment
│   ├── multi_drone_dmpc_rl_env.py    # RL variant with DMPC guide reward
│   ├── dmpc_expert.py                # Python DMPC expert (faithful reimplementation)
│   ├── dmpc_gpu_expert.py            # GPU-batched DMPC (batched ADMM)
│   ├── crazyflie.py                  # Crazyflie quadrotor asset and controller
│   ├── quadcopter_env.py             # Single-drone base environment
│   └── agents/                       # RSL-RL agent configs
├── PolicyFlow/
│   ├── policyflow_torch/             # PolicyFlow package (PyTorch)
│   │   ├── agents/                   # PPO and PolicyFlow agents
│   │   ├── modules/                  # Networks, flow, neighbor encoder, normalizer
│   │   └── runners/                  # IsaacLab training runners
│   └── scripts/isaaclab/
│       ├── train_rl_dmpc.py          # PPO-DMPC: PolicyFlow + DMPC guide (main training script)
│       ├── train_rl_ippo.py          # Vanilla PPO baseline (no DMPC guide)
│       ├── online_bc_dmpc.py         # Online behavioral cloning from DMPC expert
│       ├── offline_bc_dmpc.py        # Offline BC from collected data
│       ├── collect_dmpc_data.py      # Collect DMPC expert trajectories
│       ├── rl_model_test.py          # Evaluate trained RL policy (logs steps.csv / envs.csv)
│       ├── dmpc_expert_test.py       # Evaluate CPU/GPU DMPC expert (logs steps.csv / envs.csv)
│       ├── eval_bc_dmpc.py           # Evaluate BC model
│       └── test_bc_dmpc.py           # Evaluation with video recording
├── config/                       # Pretrained checkpoints
│   ├── phase1.pt                     # Phase 1 checkpoint (soft curriculum)
│   ├── phase2.pt                     # Phase 2 checkpoint (hard termination)
│   ├── phase3.pt                     # Phase 3 checkpoint (per-drone IPPO, full PPO-DMPC)
│   └── vanilla_ppo.pt                # Vanilla PPO baseline checkpoint
└── extras/                       # Third-party utilities
```

---

## Installation

### Python (Isaac Lab)

```bash
# Install Isaac Lab (follow official instructions: https://isaac-sim.github.io/IsaacLab)
conda activate env_isaaclab

# Install PolicyFlow package
cd PolicyFlow
pip install -e .

# Install quadcopter environments
cd ../quadcopter
pip install -e .
```

### C++ DMPC (optional, for reference CPU solver)

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j4
```
Requires: CMake >= 3.0, Eigen, qpOASES (included as submodule).

---

## Training

Training proceeds in three phases, all using `train_rl_dmpc.py` (PolicyFlow actor + DMPC velocity-alignment reward).
Each phase resumes from the previous checkpoint so that learned behaviors are preserved across transitions.

### Phase 1: Soft curriculum

Soft collision/OOB penalties; no hard episode termination. Policy learns basic goal-directed flight.

```bash
python PolicyFlow/scripts/isaaclab/train_rl_dmpc.py \
    --num_envs 800 \
    --num_drones 10 \
    --n_min 1 \
    --max_iterations 1800 \
    --log_dir runs/rl_dmpc
```

### Phase 2: Hard termination curriculum

Hard episode reset on collision/OOB. Resume from Phase 1 checkpoint.

```bash
python PolicyFlow/scripts/isaaclab/train_rl_dmpc.py \
    --resume config/phase1.pt \
    --num_envs 800 --num_drones 10 --n_min 1 \
    --curriculum \
    --max_iterations 5000 \
    --log_dir runs/rl_dmpc
```

### Phase 3: Per-drone IPPO critic

Switches to per-drone independent critic for better credit assignment. Resume from Phase 2 checkpoint.

```bash
python PolicyFlow/scripts/isaaclab/train_rl_dmpc.py \
    --resume config/phase2.pt \
    --num_envs 800 --num_drones 10 --n_min 1 \
    --curriculum --curriculum2 \
    --max_iterations 2000 \
    --log_dir runs/rl_dmpc
```

### Vanilla PPO baseline

Vanilla PPO with distance-to-goal and sparse success rewards only. No DMPC guide.
Trains independently of the phases above for ~2900 iterations.

```bash
python PolicyFlow/scripts/isaaclab/train_rl_ippo.py \
    --num_envs 800 \
    --num_drones 10 \
    --n_min 1 \
    --max_iterations 2900 \
    --log_dir runs/rl_ippo
```

---

## Evaluation

### RL policy

```bash
python PolicyFlow/scripts/isaaclab/rl_model_test.py \
    --checkpoint config/phase3.pt \
    --num_envs 1000 \
    --num_drones 10
```
Outputs `steps.csv` (per-step metrics) and `envs.csv` (per-episode aggregates) under `runs/rl_model_test/`.

### DMPC expert (CPU or GPU)

```bash
python PolicyFlow/scripts/isaaclab/dmpc_expert_test.py \
    --mode gpu \
    --num_envs 1000 \
    --num_drones 10
```
`--mode cpu` runs the original multi-threaded OSQP solver; `--mode gpu` runs the batched ADMM solver.
Outputs `steps.csv` and `envs.csv` under `runs/dmpc_expert_test/`.

---

## Observation Space

Each drone observes a vector $o^i \in \mathbb{R}^{37 + 9(N-1)}$:

| Component | Dim | Frame |
|---|---|---|
| Relative goal position | 3 | World |
| Linear velocity | 3 | World |
| Body-to-world rotation matrix | 9 | — |
| Angular velocity | 3 | Body |
| Past 5-step own positions (relative to current) | 15 | World |
| Sinusoidal episode-progress embedding | 4 | — |
| Per neighbor: rel. pos + rel. vel + neighbor goal dir | 9×(N−1) | World |

---

## Action Space

Each drone outputs $a^i \in [-1,1]^3$: a normalized desired velocity in the
**goal-aligned frame** (x toward goal, z up, y right-hand).
A geometric low-level controller converts this to motor thrust commands at 100 Hz.
The policy runs at 20 Hz.

---

## References

- L. E. Luis et al., "Online Trajectory Generation with Distributed Model Predictive Control for Multi-Robot Motion Planning," IEEE RA-L 2020.
- B. Stellato et al., "OSQP: An Operator Splitting Solver for Quadratic Programs," Mathematical Programming Computation 2020.
- C. A. de Witt et al., "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?", 2020.
- J. Hilton et al., "Batch Size-Invariance for Policy Optimization," 2022.
- T. Huang et al., "QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning," 2023.
- D. Mellinger and V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA 2011.
