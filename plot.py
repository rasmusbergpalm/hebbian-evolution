import gym
import matplotlib.pyplot as plt

from gmm_hebbian_population import GMMHebbianPopulation
from hebbian_agent import HebbianAgent
import os
import shutil

from normal_hebbian_population import NormalHebbianPopulation

num_learning_rules = 2
scale = 0.1
population = NormalHebbianPopulation(scale)
population.load("7bae779.t")
"""
for k, ml in population.mixing_logits_tensors.items():
    plt.figure()
    plt.imshow(ml.detach().numpy().argmax(axis=-1))
    plt.title(k)
    plt.colorbar()
    plt.savefig('assignments-%s.png' % k)
    plt.close()
"""

agent: HebbianAgent = population.sample(2)[0]

env = gym.make("LunarLander-v2")
obs = env.reset()
done = False
r_tot = 0

shutil.rmtree("./plots", ignore_errors=True)
os.mkdir("plots")
i = 0
rsum = 0
while not done:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={
        'width_ratios': [0.6, 0.2, 0.2]
    })
    for (k, w), ax in zip(agent.get_weights().items(), [ax2, ax3]):
        _w = w.detach().numpy()
        if _w.shape[0] < _w.shape[1]:
            _w = _w.T
        ax.imshow(_w)
        ax.axis('off')

    scene = env.render(mode='rgb_array')
    ax1.imshow(scene)
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig('plots/%03d.png' % i, bbox_inches='tight')
    plt.close()

    action = agent.action(obs)
    obs, r, done, info = env.step(action)
    rsum += r

    i += 1
print(rsum)
