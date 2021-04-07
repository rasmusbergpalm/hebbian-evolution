from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase, Ant


class CustomAnt(Ant):
    def __init__(self, morphology_xml):
        WalkerBase.__init__(self, morphology_xml, "torso", action_dim=8, obs_dim=28, power=2.5)


class CustomAntBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, morphology_xml, render=False):
        self.robot = CustomAnt(morphology_xml)
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
