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






CylinderEnv2_False.py is an environment (Env) that I added in the gym environment.

If you have any other questions, contact the author at 1713241498@qq.com.
