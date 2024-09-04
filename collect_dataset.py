import pyrallis
import minari
import gymnasium
import stable_baselines3 as sb3

from dataclasses import dataclass
import table

@dataclass
class Config:
    env: str
    dataset: str
    algo: str = 'sac'
    num_steps_per_iteration: int = 10000

class DummyTableAgent:
    def __init__(self, env):
        self.env = env
        self._dx = 1.0
        self._dy = 1.0
        self._timestep = 0

    def learn(self, **kwargs):
        # Do nothing
        pass

    def predict(self, state):
        import numpy as np
        import random

        # See if we change direction
        self._timestep += 1

        if self._timestep > 300:
            self._timestep = 0
            self._dx = 1.0 if (random.random() < 0.5) else -1.0
            self._dy = 1.0 if (random.random() < 0.5) else -1.0

        # Produce the action
        return np.array([self._dx, self._dy], dtype=np.float32), []


@pyrallis.wrap()
def main(config: Config):
    env = gymnasium.make(config.env)

    if config.algo == 'sac':
        agent = sb3.SAC(
            'MlpPolicy',
            env=env,
            verbose=1
        )
    elif config.algo == 'ppo':
        agent = sb3.PPO(
            'MlpPolicy',
            env=env,
            verbose=1
        )
    elif config.algo == 'dummy':
        agent = DummyTableAgent(env)

    print("Training an RL agent to have some decent exploration")
    env = minari.DataCollectorV0(env)

    try:
        while True:
            agent.learn(total_timesteps=5_000)

            print(" Learned for 10K steps. Collecting data")
            state, info = env.reset()
            ret = 0.0

            for _ in range(config.num_steps_per_iteration):
                action, states_ = agent.predict(state)
                state, rew, terminated, truncated, info = env.step(action)
                ret += rew

                if terminated or truncated:
                    print('R', ret)
                    env.reset()
                    ret = 0.0
    except KeyboardInterrupt:
        pass

    # We have collected data, save it now
    dataset = minari.create_dataset_from_collector_env(dataset_id=config.dataset, collector_env=env)

if __name__ == '__main__':
    main()
