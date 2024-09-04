import gymnasium

import math
import random
import sys
import numpy as np

class Table(gymnasium.Env):
    GOAL = np.array([0.5, 0.5])
    TOLERANCE = np.array([0.05, 0.05])

    def __init__(self):
        super().__init__()

        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )   # dx, dy
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )   # X, Y

    def reset(self, **kwargs):
        """ Reset the environment and return the initial state number
        """
        self._x = random.random()
        self._y = random.random()
        self._timestep = 0

        return self.current_state(), {}

    def step(self, action):
        dx, dy = action
        dx *= 0.01
        dy *= 0.01

        self._timestep += 1

        # Don't let the agent leave the room
        original_x = self._x
        original_y = self._y

        self._x += dx + 0.005 * random.random()
        self._y += dy + 0.005 * random.random()

        reward = 0.0
        terminal = False

        if not (0.0 < self._x < 1.0 and 0.0 < self._y < 1.0):
            self._x = original_x
            self._y = original_y
            reward = -50.0
            terminal = True

        # Check goal
        state = self.current_state()

        if (np.abs(state - self.GOAL) < self.TOLERANCE).all():
            reward = 100.0
            terminal = True

        # Return the current state, a reward and whether the episode terminates
        return state, reward, terminal, self._timestep > 100, {}

    def current_state(self):
        return np.array([self._x, self._y], dtype=np.float32)

gymnasium.register('Table-v1', 'table:Table')
gymnasium.register('Compressors-v1', 'flandersmake_compressors.gym_env:CompressorEnv')

import sys
import os

sys.path.append(os.path.abspath(os.curdir + '/flandersmake_compressors/'))
