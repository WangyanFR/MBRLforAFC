import numpy as np
"""
Utilities for online normalization of states and rewards.

- RunningMeanStd: keeps a running estimate of mean and standard deviation.
- Normalization:   normalizes input states.
- RewardScaling:   normalizes discounted returns to stabilize training.
"""

class RunningMeanStd:
    """Maintain running estimates of mean and standard deviation."""
    def __init__(self, shape):  # shape:the dimension of input data
        """
        Args:
            shape (int or tuple): shape of the incoming data (e.g., state dimension).
        """
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        """Update running mean and std with a new sample x."""
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            # For the very first sample, set mean and std directly
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            # incremental update for mean
            self.mean = old_mean + (x - old_mean) / self.n
            # Welford-like update for variance accumulator
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """Callable state normalizer based on RunningMeanStd."""
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        """
        Normalize input x using running mean and std.

        Args:
            x (np.ndarray): input vector (e.g., state).
            update (bool): whether to update running statistics.
                           During evaluation, set update=False.

        Returns:
            np.ndarray: normalized x.
        """
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    """
    Maintain a discounted return R_t and normalize it using RunningMeanStd.

    This tends to reduce non-stationarity of the reward signal.
    """
    def __init__(self, shape, gamma):
        """
        Args:
            shape (int or tuple): shape of reward (usually 1).
            gamma (float): discount factor.
        """
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        """
        Scale incoming reward.

        Args:
            reward (float or np.ndarray): the immediate reward.

        Returns:
            np.ndarray: scaled reward.
        """
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        """Reset discounted return R. Call at the beginning of each episode."""
        self.R = np.zeros(self.shape)
