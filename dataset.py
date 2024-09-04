import gymnasium
import numpy as np

class Dataset:
    def __init__(self, name):
        # Load
        print("Loading dataset", name)

        if name.endswith('.mat'):
            self.load_matlab(name)
        else:
            self.load_minari(name)

        self.obs = np.concatenate(self.obs).astype(np.float32)
        self.next_obs = np.concatenate(self.next_obs).astype(np.float32)
        self.actions = np.concatenate(self.actions).astype(np.float32)
        self.rewards = np.concatenate(self.rewards).astype(np.float32)[:, None]
        self.dones = np.concatenate(self.dones)               # Keep shape (N,)
        self.has_next = ~self.dones

    def load_matlab(self, name):
        import scipy.io

        dataset = scipy.io.loadmat(name, simplify_cells=True)['dataset']

        # Keys
        kstates = []
        kactions = []
        krewards = []

        for k in dataset[0].keys():
            if 'Reward' in k:
                krewards.append(k)
            if 'State' in k:
                kstates.append(k)
            if 'Action' in k:
                kactions.append(k)

        # Concatenate the episodes in the dataset
        print("Concatenating episodes")
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []

        for episode in dataset:
            ep_obs = np.concatenate([episode[k][:, None] for k in kstates], axis=1)
            ep_act = np.concatenate([episode[k][:, None] for k in kactions], axis=1)
            ep_rew = np.concatenate([episode[k][:, None] for k in krewards], axis=1).sum(1)
            ep_done = np.zeros_like(ep_rew, dtype=np.bool_)
            ep_done[-2] = True  # The last one (-1) will be removed

            self.obs.append(ep_obs[:-1])
            self.next_obs.append(ep_obs[1:])
            self.actions.append(ep_act[:-1])
            self.rewards.append(ep_rew[:-1])
            self.dones.append(ep_done[:-1])
            print(self.dones)

    def load_minari(self, name):
        import minari

        dataset = minari.load_dataset(name)

        # Concatenate the episodes in the dataset
        print("Concatenating episodes")
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []

        for index, episode in enumerate(dataset.iterate_episodes()):
            ep_obs = episode.observations

            if isinstance(ep_obs, dict):
                ep_obs = ep_obs['observation']

            self.obs.append(ep_obs[:-1])
            self.next_obs.append(ep_obs[1:])
            self.actions.append(episode.actions)
            self.rewards.append(episode.rewards)
            self.dones.append(episode.terminations | episode.truncations)

class EmptyDataset:
    """ Dataset with Numpy arrays that can be appended to, but does not load data from anywhere
    """
    def __init__(self, env):
        def shape_for_space(space):
            if isinstance(space, gymnasium.spaces.Discrete):
                return (space.n,)
            else:
                return space.shape

        state_shape = shape_for_space(env.observation_space)
        action_shape = shape_for_space(env.action_space)

        self.obs = np.zeros((0,) + state_shape, dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((0,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((0, 1), dtype=np.float32)
        self.dones = np.zeros((0,), dtype=np.bool_)
        self.has_next = np.zeros_like(self.dones)

    def add(self, obs, next_obs, action, reward, done):
        def append(arr, v):
            return np.append(arr, v[None, :], axis=0)

        self.obs = append(self.obs, obs)
        self.next_obs = append(self.next_obs, next_obs)
        self.actions = append(self.actions, action)
        self.rewards = append(self.rewards, np.array([reward], dtype=np.float32))
        self.dones = np.append(self.dones, done)
        self.has_next = np.append(self.has_next, not done)

