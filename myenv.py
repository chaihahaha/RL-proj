import pyrobolearn as prl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.rewards.terminal_rewards import TerminalReward
from pyrobolearn.states.body_states import PositionState, VelocityState
from pyrobolearn.states import BasePositionState, JointPositionState, JointVelocityState
from pyrobolearn.actions.robot_actions.joint_actions import JointPositionAction
from pyrobolearn.tasks.reinforcement import RLTask
from pyrobolearn.terminal_conditions.terminal_condition import TerminalCondition

N_STATES = 36
N_ACTIONS = 15
MEMORY_CAPACITY = 1000
EPSILON = 0.9 
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
STATES_SHAPE = [3,15,15,3]
LR = 1e-3
BATCH_SIZE = 20
class HasPickedAndPlacedCondition(TerminalCondition):
    def __init__(self, robot, box, world):
        self.robot = robot
        self.box = box
        self.world = world
    def check(self):
        x,y,z = self.world.get_body_position(self.box)
        return z>0.5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
        
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(N_ACTIONS, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        q = self.out(x)
        return q


class DQN(object):
    def __init__(self, actions):
        self.actions = actions
        self.eval_net, self.target_net, self.qnet = Net(), Net(), QNet()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 1))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def predict(self, x, to_numpy):
        x = torch.FloatTensor(x)
        x = torch.unsqueeze(x, 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            action = self.eval_net.forward(x).detach().numpy().reshape(N_ACTIONS)
        else:   # random
            action = self.actions.space.sample()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+N_ACTIONS].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+N_ACTIONS:N_STATES+N_ACTIONS+1])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.qnet(self.eval_net(b_s))  # shape (batch, 1)
        q_next = self.qnet(self.target_net(b_s_).detach())     # detach from graph, don't backpropagate
        
        q_target = b_r + GAMMA * q_next   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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

    def __init__(self, state, action, rate=1, seed=None, *args, **kwargs):
        model = DQN(action)
        super(RandomPolicy, self).__init__(state, action, model=model, rate=rate, *args, **kwargs)

    def sample(self, state=None):
        return self.act(state)

if __name__=="__main__":
    sim = prl.simulators.Bullet()
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    states = BasePositionState(manipulator) + JointPositionState(manipulator) + JointVelocityState(manipulator) + PositionState(box,world)
    action = JointPositionAction(manipulator)
    policy = RandomPolicy(states,action)
    reward = TerminalReward(HasPickedAndPlacedCondition(manipulator,box,world))
    env = Env(world, states, rewards=reward,actions=action)
    
    
    num_steps=10000
    env.reset()
    env.render()
    for t in count():
        if t>=num_steps:
            break
        obs = env.state.vec_data
        action = policy.act(obs)
        obs_, reward, done, info = env.step(action)
        print(reward)
        policy.model.store_transition(obs, action, reward, obs_)
        if policy.model.memory_counter > MEMORY_CAPACITY:
            policy.model.learn()
