from typing import Dict
import numpy as np
import torch as t
from evostrat import NormalPopulation, GaussianMixturePopulation

from agents.ant.hebbian_ant import HebbianAnt
from agents.racer.hebbian_racer import HebbianCarRacingAgent
import envs

if __name__ == '__main__':
    device = "cuda" if t.cuda.is_available() else "cpu"
    envs = [
        {},
        # {'friction': 0.5},
        # {'friction': 2.0},
        # {'side_force': -10.0},
        # {'side_force': 10.0}
    ]
    test_env = None
    agent = HebbianAnt
    param_shapes = agent.param_shapes()
    params = t.load('../8231ab7/best.t', map_location=t.device('cpu'))


    def constructor(params: Dict):
        params = {k: p.detach().to("cpu") for k, p in params.items()}
        return agent(params, test_env)


    n_rules = 1  # int(sum([t.Size(s).numel() for s in param_shapes.values()]) / rho)
    population = GaussianMixturePopulation({k: t.Size(v[:-1]) for k, v in param_shapes.items()}, (n_rules, 5), constructor, 0.1, device)
    for k, p in zip(population.mixing_logits.keys(), params[:-1]):
        population.mixing_logits[k] = p
    population.component_means = params[-1]

    for env in envs:
        test_env = env
        fits = [ind.fitness() for ind, logp in population.sample(100)]
        print(test_env, np.mean(fits), np.std(fits))
