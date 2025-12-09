import numpy as np
import random
from model_train import NN_pred


class FakeEnv():

    def __init__(self):
        self.episode_number = 0

    def reset(self):
        if self.episode_number > 0:
            chance = random.random()
            probability_hard_reset = 0.2
            if chance < probability_hard_reset:
                next_state = np.loadtxt('./Re1000_initial_flowfield.txt')
            else:
                next_state = self.next_state
            self.episode_number += 1
        else:
            self.episode_number += 1
            next_state = np.loadtxt('./Re1000_initial_flowfield.txt')

        return next_state

    def step(self,state,action,episode_steps):
        input = np.concatenate((state, action), axis=-1)
        output = NN_pred(input)
        self.reward = output[:1, ]
        self.next_state = output[1:, ]
        self.next_state = self.next_state + state
        done = False
        if episode_steps == 40:
            done = True
        truncated = False
        return self.next_state, self.reward, done, truncated, {}

    def close(self):
        pass

    def seed(self,seed):
        return None






