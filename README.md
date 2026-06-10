# Online DMPC to Learning

Multi-drone navigation via GPU-batched Distributed MPC (DMPC) and reinforcement learning.
A PolicyFlow actor (conditional rectified flow) is trained with DMPC velocity-alignment reward in Isaac Lab,
enabling decentralized collision-aware navigation without the DMPC solver at inference.

Code: https://github.com/Yong-m/online_dmpc_to_learning

---

## Repository Structure

```
online_dmpc_to_learning/
├── cpp/                  # C++ reference DMPC implementation (Luis et al. 2020)
├── matlab/               # Visualization and benchmarking scripts
├── quadcopter/           # Isaac Lab environments and GPU DMPC expert
│   ├── multi_drone_dmpc_env.py       # Base multi-drone environment
│   ├── multi_drone_dmpc_rl_env.py    # RL variant with DMPC guide reward
│   ├── dmpc_expert.py                # Python DMPC expert (faithful reimplementation)
│   └── dmpc_gpu_expert.py            # GPU-batched DMPC (ADMM)
├── PolicyFlow/
│   ├── policyflow/                   # PolicyFlow package (PyTorch)
│   │   └── policyflow_torch/
│   │       ├── agents/               # PPO and PolicyFlow agents
│   │       ├── modules/              # Networks, flow, neighbor encoder, normalizer
│   │       └── runners/              # IsaacLab and Gym training runners
│   └── scripts/isaaclab/
│       ├── train_rl_dmpc.py          # PolicyFlow + DMPC guide (main training script)
│       ├── train_rl_ippo.py          # Vanilla PPO baseline (no DMPC guide)
│       ├── online_bc_dmpc.py         # Online behavioral cloning from DMPC expert
│       ├── offline_bc_dmpc.py        # Offline BC from collected data
│       ├── collect_dmpc_data.py      # Collect DMPC expert trajectories
│       └── test_bc_dmpc.py           # Evaluation and video recording
├── config/               # Pretrained checkpoints
│   ├── phase1.pt             # Phase 1 checkpoint (soft curriculum, PolicyFlow + DMPC guide)
│   └── phase2.pt             # Phase 2 checkpoint (hard termination curriculum)
└── extras/               # Third-party utilities
```

---

## Installation

### Python (Isaac Lab)

```bash
# Install Isaac Lab (follow official instructions: https://isaaclab.ai)

# Install PolicyFlow package
cd PolicyFlow
pip install -e .

# Install quadcopter environments
cd ../quadcopter
pip install -e .
```

### C++ DMPC (optional, for reference solver)

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j4
```
Requires: CMake >= 3.0, Eigen, qpOASES (included as submodule).

---

## Training

### Phase 1–2: PolicyFlow with DMPC Guide

Trains a conditional rectified flow actor guided by a GPU-batched DMPC expert.
`--curriculum` enables soft collision/OOB penalties (Phase 1).
`--curriculum2` enables hard termination on collision/OOB (Phase 2).

```bash
python PolicyFlow/scripts/isaaclab/train_rl_dmpc.py \
    --num_envs 800 \
    --num_drones 10 \
    --n_min 1 \
    --curriculum2 \
    --max_iterations 500000 \
    --log_dir runs/rl_dmpc
```

Resume from a pretrained checkpoint (e.g., provided Phase 1/2 checkpoints):
```bash
# Resume Phase 2 from the provided Phase 1 checkpoint
python PolicyFlow/scripts/isaaclab/train_rl_dmpc.py \
    --resume config/phase1.pt \
    --curriculum2 \
    --num_envs 800 --num_drones 10

# Resume Phase 3 (IPPO) from the provided Phase 2 checkpoint
python PolicyFlow/scripts/isaaclab/train_rl_ippo.py \
    --resume config/phase2.pt \
    --num_envs 800 --num_drones 10
```

### Phase 3: Independent PPO (IPPO, no DMPC guide)

Vanilla PPO baseline or Phase 3 fine-tuning. No DMPC expert — no solver bottleneck.

```bash
python PolicyFlow/scripts/isaaclab/train_rl_ippo.py \
    --num_envs 800 \
    --num_drones 10 \
    --n_min 1 \
    --max_iterations 500000 \
    --log_dir runs/rl_ippo
```

### Online Behavioral Cloning

```bash
python PolicyFlow/scripts/isaaclab/online_bc_dmpc.py \
    --num_envs 32 \
    --num_drones 4 \
    --save_path runs/online_bc/model.pt
```

---

## Evaluation

```bash
python PolicyFlow/scripts/isaaclab/test_bc_dmpc.py \
    --checkpoint runs/rl_dmpc/<experiment>/checkpoints/model_best.pt \
    --num_envs 4 \
    --num_drones 4
```

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
- C. A. de Witt et al., "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?", 2020.
- J. Hilton et al., "Batch Size-Invariance for Policy Optimization," 2022.
- T. Huang et al., "QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning," 2023.
- D. Mellinger and V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA 2011.
