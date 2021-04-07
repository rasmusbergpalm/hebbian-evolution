import os
from shutil import copytree, copy

import gym
import pybullet_data
import glob
from envs.car_racing import CarRacing
from envs.custom_ant import CustomAntBulletEnv

gym.register(id='CarRacingCustom-v0',
             entry_point='envs:CarRacing',
             max_episode_steps=1000,
             reward_threshold=900)

# Monkeypatch the custom ant morphologies into pybullet
our_xml_dir = os.path.join(os.path.dirname(__file__), 'ants')
pybullet_xml_dir = os.path.join(pybullet_data.getDataPath(), "mjcf")
for f in glob.glob(our_xml_dir + '/*.xml'):
    copy(f, pybullet_xml_dir)

gym.register(id='CustomAntBulletEnv-v0',
             entry_point='envs:CustomAntBulletEnv',
             max_episode_steps=1000,
             reward_threshold=2500.0)
