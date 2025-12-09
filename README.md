# MBRLforAFC
# Flow Control with PPO: MFRL, MBRL and MBTT

This repository implements **reinforcement learning–based flow control** using Proximal Policy Optimization (PPO) in three flavors:

- **MFRL** – Model-Free Reinforcement Learning
PPO interacts directly with an ANSYS Fluent–based CFD environment.

- **MBRL** – Model-Based Reinforcement Learning
A neural network surrogate model of the environment is trained from CFD data and then used as a fast “virtual environment”.

- **MBTT** – Model-Based Two-Task / Transfer Training
A model trained at one Reynolds number (e.g. Re=1000) is adapted to another (e.g. Re=100) and then used for RL.

---

## 1.  Environment & Dependencies

### 1.1 System Requirements

- **OS**: Windows 10 / 11
- **Python**: 3.8 – 3.10 (recommended)
- **CFD Solver**: ANSYS Fluent (licensed and callable from Python)
- **Hardware**:
- CPU: multi-core recommended
- GPU: NVIDIA GPU recommended (for faster NN training, optional)

### 1.2 Python Dependencies

Main Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch` (PyTorch)
- `gym`
- `ansys-fluent-core` (Python interface to Fluent)


## 2. Code Structure
### 2.1 File Overview

| File                     | Description                                                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `normalization.py`       | Utilities for **running mean/std**, **state normalization** and **reward scaling** (e.g. `RunningMeanStd`, `Normalization`, `RewardScaling`) used to stabilize PPO training.                                             |
| `replaybuffer.py`        | On-policy **replay buffer** for PPO, storing transitions `(s, a, logprob, r, s', dw, done)` and triggering updates when full.                                                                                            |
| `ppo_continuous.py`      | Core **PPO implementation** for continuous action spaces, including: Actor (Beta / Gaussian policy), Critic (value network), GAE, entropy regularization, gradient clipping, learning-rate decay, etc.                   |
| `CylinderEnv2_False.py`  | **Real CFD environment** built on ANSYS Fluent (2D cylinder flow control) with Gym-style `reset()` / `step()`; handles boundary conditions, Fluent runs, state extraction, and reward computation.                       |
| `PPO_continuous_main.py` | Main training script for **MFRL**: runs PPO directly on the Fluent-based environment `CylinderEnv2`.                                                                                                                     |
| `model_train.py`         | Trains a **neural network environment model** using Re = 1000 CFD data. Builds inputs `[state(151), action(1)]` and labels `[reward(1), delta_state(151)]`, and provides `NN_pred()` to predict `[reward, delta_state]`. |
| `FakeEnv.py`             | Wraps `NN_pred()` into a **virtual environment** (`FakeEnv`) with Gym-like interface (`reset`, `step`) for **MBRL / MBTT** training.                                                                                     |
| `model_train_Re100.py`   | Trains / fine-tunes a **Re = 100** environment model using Re = 100 dataset (e.g. `.npy` files), for **transfer learning (MBTT)**.                                                                                       |

### 2.2 Conceptual Data Flow

Real CFD (Fluent)
    │
    ├─► MFRL: PPO_continuous_main.py + CylinderEnv2_False.py
    │
Offline CFD Data (Re=1000 / Re=100)
    │
    ├─► model_train.py           (Re=1000 model)
    ├─► model_train_Re100.py     (Re=100 model / fine-tuning)
    │
    └─► NN_pred()  ─► FakeEnv.py ─► PPO (MBRL / MBTT)

## MFRL – Model-Free Reinforcement Learning

In MFRL, PPO interacts directly with the real CFD environment. This is the most physically accurate but also the most expensive training mode (since Fluent must run for each step).


## MBRL – Model-Based Reinforcement Learning

In MBRL, we first learn a surrogate NN model of the environment from CFD data, then train PPO on this model using FakeEnv. This is much faster than MFRL since it avoids calling Fluent at every step.

MBRL has two stages:

Offline model training from CFD data (e.g. Re = 1000)

Online PPO training on the learned model (FakeEnv)

### Stage 1 – Train the NN Environment Model (Re = 1000)

Prepare a dataset folder, e.g.:

./Re1000data/
    flowfield/episode_*.csv
    cdcl.csv
    (other required files)

### Stage 2 – PPO on the Learned Model (FakeEnv)

FakeEnv.py uses NN_pred to build a virtual environment:



## Citation

If you use this code for academic work, please cite the corresponding paper (example):

@article{your_paper_2025,
  title   = {},
  author  = {},
  journal = {},
  year    = {2025}
}

## Contact

For questions, bug reports, or collaboration:

Open an issue in this repository, or

Contact:  1713241498@qq.com / 1022201147@tju.edu.cn

