import pyrobolearn as prl
import numpy as np
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.rewards.terminal_rewards import TerminalReward
from pyrobolearn.states.body_states import PositionState, VelocityState
from pyrobolearn.states import BasePositionState, JointPositionState, JointVelocityState
from pyrobolearn.actions.robot_actions.joint_actions import JointPositionAction
from pyrobolearn.tasks.reinforcement import RLTask
from pyrobolearn.terminal_conditions.terminal_condition import TerminalCondition

class HasPickedAndPlacedCondition(TerminalCondition):
    def __init__(self, robot, box):
        self.robot = robot
        self.box = box
    def check(self):
        return False

class RandomPolicy(Policy):
    class RandomModel(object):
        def __init__(self, actions, seed=None):
            self.seed = seed
            self.actions = actions
        @property
        def seed(self):
            return self._seed
        @seed.setter
        def seed(self, seed):
            if seed is not None:
                np.random.seed(seed)
            self._seed = seed
        def reset(self):
            pass

        def predict(self, state=None, to_numpy=True):
            space = self.actions.space
            return space.sample() 

    def __init__(self, state, action, rate=1, seed=None, preprocessors=None, postprocessors=None, *args, **kwargs):
        model = self.RandomModel(action, seed=seed)
        super(RandomPolicy, self).__init__(state, action, model=model, rate=rate, preprocessors=preprocessors, postprocessors=postprocessors, *args, **kwargs)

    def sample(self, state=None):
        return self.act(state)

if __name__=="__main__":
    sim = prl.simulators.Bullet()
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    states = BasePositionState(manipulator) + JointPositionState(manipulator) + JointVelocityState(manipulator) + PositionState(box,world) + VelocityState(box,world)
    action = JointPositionAction(manipulator)
    policy = RandomPolicy(states,action)
    reward = TerminalReward(HasPickedAndPlacedCondition(manipulator,box))
    env = Env(world, states, rewards=reward,actions=action)
    task = RLTask(env, policy)
    task.run(num_steps=10000,render=True)
