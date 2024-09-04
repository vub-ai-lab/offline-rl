import dataset
import sys
import os
import copy
from concurrent.futures.process import ProcessPoolExecutor

from dataclasses import dataclass
import pyrallis
import torch
import numpy as np
import numba as nb
import tqdm

import pygad
import pygad.torchga

jit = nb.njit(fastmath=True, cache=True)

def tostr(x):
    return ' '.join([str(float(i)) for i in x.flatten()])

@dataclass
class Config:
    dataset: str

    num_policies: int
    num_state_clusters: int

    hidden: int = 128
    layers: int = 1
    lr: float = 0.001
    vi_iterations: int = 100

    jobs: int = 4

@jit
def value_iteration(qtable, r, NV, gamma, lr):
    for state in range(1, qtable.shape[0]):                                     # NOTE: Starts at 1. We keep the Q-Values for state 0 ("done") to 0.
        for action in range(qtable.shape[1]):
            target_qvalue = r[state, action] + gamma * NV[state, action]
            qtable[state, action] += lr * (target_qvalue - qtable[state, action])

@jit
def build_mdp(clusters, next_clusters, actions, rewards, dones, R, T, Mu):
    N = clusters.shape[0]
    is_first = True

    for experience_index in range(N):
        reward = rewards[experience_index]
        done = dones[experience_index]

        state = clusters[experience_index] + 1
        next_state = 0 if done else next_clusters[experience_index] + 1
        action = actions[experience_index]

        # Idea for the reward function: the reward for a policy in a cluster is the sum
        # of rewards obtained by every experience for that policy in the cluster, divided
        # by the number of times this policy "left" the cluster in the dataset (R / T.sum())
        R[state, action, next_state] += reward

        if next_state != state:
            T[state, action, next_state] += 1

        if is_first:
            Mu[state] += 1

        # Detect first experiences
        is_first = False

        if done:
            is_first = True

class DiscreteVAE(torch.nn.Module):
    def __init__(self, in_shape, out_shape, config):
        super().__init__()
        self.config = config

        def make_model(in_dim, out_dim):
            layers = []

            layers.append(torch.nn.Linear(in_dim, config.hidden))
            layers.append(torch.nn.Tanh())

            for i in range(config.layers - 1):
                layers.append(torch.nn.Linear(config.hidden, config.hidden))
                layers.append(torch.nn.Tanh())

            layers.append(torch.nn.Linear(config.hidden, out_dim))

            return torch.nn.Sequential(*layers)

        # State and action to a Softmax over policy indexes
        self.guess_policy = make_model(
            np.prod(in_shape) + np.prod(out_shape),
            config.num_policies,
        )

        # State and policy index to action
        self.guess_action = make_model(
            np.prod(in_shape) + config.num_policies,
            np.prod(out_shape)
        )

    def forward(self, states, actions):
        policy_probas = self.get_policy_probas(states, actions)
        policy_dist = torch.distributions.OneHotCategoricalStraightThrough(probs=policy_probas)

        # Sample policy indexes
        policy_onehot = policy_dist.rsample()

        return self.get_actions(states, policy_onehot)

    def get_policy_probas(self, states, actions):
        flat_states = states.view(states.shape[0], -1)
        flat_actions = actions.view(actions.shape[0], -1)

        # Guess the policy
        policy_logits = self.guess_policy(
            torch.cat([flat_states, flat_actions], dim=1)
        )

        policy_probas = torch.distributions.utils.logits_to_probs(policy_logits)

        return policy_probas

    def get_actions(self, states, policy_onehot):
        flat_states = states.view(states.shape[0], -1)

        # States and policy indexes to actions
        guessed_actions = self.guess_action(
            torch.cat([flat_states, policy_onehot], dim=1)
        )

        return guessed_actions

class OfflineRL:
    def __init__(self, config):
        self.config = config

        # File that stores the best Q-Value
        with open("best_qvalue.txt", "w") as f:
            f.write('-10000.0\n')

    def train(self):
        self.train_state_to_action()
        self.optimize_clustering()

    def train_state_to_action(self):
        # Create the neural network
        self.state_to_action = DiscreteVAE(d.obs.shape[1:], d.actions.shape[1:], self.config)

        # Train it to predict actions
        state_actions_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(d.obs),
            torch.from_numpy(d.actions)
        )

        optimizer = torch.optim.Adam(self.state_to_action.parameters(), lr=self.config.lr)
        loss = torch.nn.MSELoss()
        batch_size = 256

        train_loader = torch.utils.data.DataLoader(
            state_actions_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.state_to_action.train()

        try:
            for epoch in range(1, 1000):
                with tqdm.tqdm(train_loader, unit="batch") as tepoch:
                    for states, actions in tepoch:
                        tepoch.set_description(f"Epoch {epoch}")

                        predicted_actions = self.state_to_action(states, actions)
                        l = loss(predicted_actions, actions)

                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()

                        tepoch.set_postfix(loss=l.item())
        except KeyboardInterrupt:
            pass

    def evaluate_solution(self, sol):
        params = pygad.torchga.model_weights_as_dict(self.state_to_cluster, sol)
        self.state_to_cluster.load_state_dict(params)

        return self.solve_mdp()


    def optimize_clustering(self):
        # Neural network that maps a state to a cluster index
        in_dim = np.prod(d.obs.shape[1:])
        out_dim = self.config.num_state_clusters

        # Number of intersections of N planes is N(N-1)/2. N is the number of hidden
        # neurons, the number of intersections will define the decision boundaries
        # where the argmax over out_dim changes. We want this number to be equal
        # to the number of clusters (TODO: a better number may exist)
        #
        # N(N-1)/2 = out_dim
        # N(N-1) = 2*out_dim
        # N² - N - 2*out_dim = 0
        #
        # delta = 1 + 4*1*2*out_dim = 1 + 8*out_dim
        # roots = 1 +- sqrt(1 + 8*out_dim), we take the + root to be positive
        #
        # hidden_dim = 1 + sqrt(1 + 8*out_dim)
        hidden_dim = int(1 + np.sqrt(1 + 8 * out_dim))
        print('hidden_dim', hidden_dim)

        self.state_to_cluster = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Tanh()
        )

        # Process pool
        executor = ProcessPoolExecutor(max_workers=self.config.jobs)

        # Optimize that neural network to produce good clusterings
        def fitness_func(ga_instance, solution, sol_idx):
            rs = []

            for sol in solution:
                rs.append(executor.submit(
                    self.evaluate_solution,
                    sol
                ))

            for i in range(len(rs)):
                rs[i] = rs[i].result()

            return rs

        def on_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}")

        torch_ga = pygad.torchga.TorchGA(model=self.state_to_cluster, num_solutions=100)

        ga_instance = pygad.GA(
            num_generations=500,
            num_parents_mating=50,
            initial_population=torch_ga.population_weights,
            fitness_func=fitness_func,
            fitness_batch_size=64,
            on_generation=on_generation)

        ga_instance.run()

    def solve_mdp(self):
        """ Given a neural network that maps states to cluster indexes, compute and solve an MDP
        """
        num_states = self.config.num_state_clusters + 1   # State 0 means "terminal"
        num_actions = self.config.num_policies

        T = np.zeros((num_states, num_actions, num_states), dtype=np.float32)       # Stochastic transitions. State number 0 means "done".
        R = np.zeros((num_states, num_actions, num_states), dtype=np.float32)       # Stochastic reward function R[state, action, next_state]. The actual next_state influences the reward
        Mu = np.zeros((num_states,), dtype=np.float32)                              # Initial state distribution

        # Map experiences to their clusters, next clusters and policies
        with torch.no_grad():
            clusters = self.state_to_cluster(
                torch.from_numpy(d.obs)
            ).argmax(1).numpy()
            next_clusters = self.state_to_cluster(
                torch.from_numpy(d.next_obs)
            ).argmax(1).numpy()
            actions = self.state_to_action.get_policy_probas(
                torch.from_numpy(d.obs),
                torch.from_numpy(d.actions)
            ).argmax(1).numpy()
            reconstructed_actions = self.state_to_action(
                torch.from_numpy(d.obs),
                torch.from_numpy(d.actions)
            ).numpy()

        build_mdp(clusters, next_clusters, actions, d.rewards.ravel(), d.dones, R, T, Mu)

        # Compute the actual rewards: divide the sum of rewards from experiences by the number of times we "leave" the cluster
        s = T.sum(2, keepdims=True)

        R /= s + 1
        T /= s + 0.0001
        Mu /= Mu.sum()

        # Value iteration on the resulting MDP
        qtable = np.zeros((num_states, num_actions), dtype=np.float32)

        r = (R * T).sum(2)

        for iteration in range(self.config.vi_iterations):
            V = qtable.max(1)       # Maximum over actions for every state
            NV = (T * V[None, None, :]).sum(2)

            value_iteration(qtable, r, NV, 0.99, 0.1)

        # Cost function: average value of the states across the initial state distribution
        avg = (qtable.max(1) * Mu).sum()
        best_qvalue = None

        print('QV', avg)

        while best_qvalue is None:
            try:
                with open("best_qvalue.txt", "r") as f:
                    best_qvalue = float(f.readline())
            except ValueError:
                pass

        if avg > best_qvalue:
            print('New best average Q-Value', avg)

            with open("best_qvalue.txt", "w") as f:
                f.write(f"{avg}\n")

            with open(f"out-{avg}.txt", "w") as f:
                for i in range(d.obs.shape[0]):
                    state = int(clusters[i]) + 1
                    action = int(actions[i])
                    done = d.dones[i]
                    next_state = 0 if done else int(next_clusters[i]) + 1

                    print('S', tostr(d.obs[i]), state, action, next_state, qtable[state, action], int(qtable[state].argmax()), R[state, action, next_state], file=f)

                    if action == qtable[state].argmax():
                        print('OPT', tostr(d.obs[i]), state, action, tostr(d.actions[i]), tostr(reconstructed_actions[i]), file=f)

            torch.save({
                'state_to_cluster': self.state_to_cluster,
                'state_to_action': self.state_to_action,
                'qtable': qtable,
            }, f"agent-{avg}.pt")

        return avg

d = None

@pyrallis.wrap()
def main(config: Config):
    global d

    print('Loading dataset...')
    d = dataset.Dataset(config.dataset)

    print('Training')
    offlinerl = OfflineRL(config)
    offlinerl.train()

if __name__ == '__main__':
    main()
