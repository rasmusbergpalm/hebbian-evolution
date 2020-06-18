import random
from typing import Callable, Iterable

import torch as t
import torch.nn as nn
import gym
import tqdm as tqdm


def identity(x):
    return x


class HebbianLayer(nn.Module):
    @staticmethod
    def random(n_in, n_out, activation_fn=identity):
        return HebbianLayer(t.randn(n_in, n_out), t.randn(n_in, n_out), t.randn(n_in, ), t.randn(n_out, ), t.randn(n_in, n_out), t.randn(n_in, n_out), activation_fn)

    def __init__(self, W_init: t.Tensor, A: t.Tensor, B: t.Tensor, C: t.Tensor, D: t.Tensor, eta: t.Tensor, activation_fn):
        super().__init__()
        self.W_init = W_init
        self.W = W_init + 0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.eta = eta
        self.activation_fn = activation_fn

    def get_params(self):
        return self.W_init, self.A, self.B, self.C, self.D, self.eta

    def forward(self, pre):
        post = self.activation_fn(pre @ self.W)
        self.update(pre, post)
        return post

    def update(self, pre, post):
        self.W += self.eta * (self.A * (pre[:, None] @ post[None, :]) + self.B @ pre + self.C @ post + self.D)


class HebbianAgent:
    def __init__(self, W, A, B, C, D, eta):
        self.net = nn.Sequential(
            HebbianLayer(W, A, B, C, D, eta, t.nn.Softmax(dim=0)),
        )

    def run(self, env):
        obs = env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs)
            obs, r, done, info = env.step(action)
            r_tot += r

        return r_tot

    def action(self, obs):
        with t.no_grad():
            return t.argmax(self.net(t.tensor(obs, dtype=t.float32))).item()


def evolve(fitness_fn: Callable[[Iterable[t.Tensor]], float], initial: Iterable[t.Tensor], iterations: int, pop_size: int, noise: float, learning_rate: float):
    eps = 1e-6
    best = initial
    best_fit = fitness_fn(initial)
    pbar = tqdm.tqdm(range(iterations))
    for _ in pbar:
        population = [tuple(t.randn(p.shape) for p in best) for _ in range(pop_size)]  # (pop_size, (*param_size))
        try_pop = [tuple(b + noise * p for b, p in zip(best, ind)) for ind in population]
        fitness = t.tensor([fitness_fn(ind) for ind in try_pop])  # (pop_size, )
        best_fit = max(fitness.max().item(), best_fit)
        pbar.set_description("best %0.3f" % best_fit)
        norm_fitness = (fitness - fitness.mean()) / (eps + fitness.std())  # (pop_size, )
        fitness_weighted_sum_pop = tuple(sum([fit * ind[i] for ind, fit in zip(population, norm_fitness)]) for i in range(len(population[0])))
        best = tuple(bp + learning_rate / (pop_size * noise) * fwp for bp, fwp in zip(best, fitness_weighted_sum_pop))

    return best, best_fit


if __name__ == '__main__':
    env = gym.make("CartPole-v0")


    def fitness_fn(params: Iterable[t.Tensor]):
        agent = HebbianAgent(*params)
        return agent.run(env)


    initial = HebbianLayer.random(4, 2).get_params()
    best, best_fit = evolve(fitness_fn, initial, 100, 200, 0.1, 0.2)
    print(best_fit)
