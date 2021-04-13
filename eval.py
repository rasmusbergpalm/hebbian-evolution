from typing import Dict

import numpy as np

import torch as t
from evostrat import NormalPopulation

from agents.racer.static_racer import StaticCarRacingAgent

import envs


if __name__ == '__main__':
    device = "cuda" if t.cuda.is_available() else "cpu"

    envs = [
        {},
        {'friction': 0.5},
        {'friction': 2.0},
        {'side_force': -10.0},
        {'side_force': 10.0}
    ]

    test_env = None
    agent = StaticCarRacingAgent
    param_shapes = agent.param_shapes()
    params = t.load('../b787e93/best.t', map_location=t.device('cpu'))


    def constructor(params: Dict):
        params = {k: p.detach().to("cpu") for k, p in params.items()}
        return agent(params, test_env)


    population = NormalPopulation(param_shapes, constructor, 0.1*0.999**300, True)
    for k, p in zip(population.param_means.keys(), params):
        population.param_means[k] = p

    for env in envs:
        test_env = env
        fits = [ind.fitness() for ind, logp in population.sample(100)]
        print(test_env, np.mean(fits), np.std(fits))
