# MBRLforAFC
Flow Control via PPO: MFRL / MBRL / MBTT

This project implements flow field control based on reinforcement learning (RL) methods, including:

MFRL (Model-Free Reinforcement Learning) - Interact directly with the Fluent real environment;

MBRL (Model-based Reinforcement Learning) - First train the neural network environment Model with data, and then train it in a virtual environment;

MBTT (Model-Based Two-Task/Transfer) - Model-based reinforcement learning that transfers between different Reynolds numbers (Re=100, Re=1000).


Environmental Requirements (Environment)
Operating system Windows 10/11 (ANSYS Fluent needs to be installed)
Python versions 3.8 to 3.10
It is recommended to use NVIDIA GPU (optional).
Dependent libraries: numpy, pandas, matplotlib, scikit-learn, torch, gym, ansys-fluent core






CylinderEnv2_False.py is an environment (Env) that I added in the gym environment.

If you have any other questions, contact the author at 1713241498@qq.com.
