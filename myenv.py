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
γ = 0.9
αw = 1e-3
αθ = 1e-3
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
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = 4 * self.tanh(self.out(x))
        return actions_value
        
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc2 = nn.Linear(N_ACTIONS, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = F.relu(torch.stack([x1,x2],1).view(-1,100))
        q = self.out(x)
        q = self.tanh(q)
        return q

# Deterministic Actor Critic
class DAC(Policy):
    def __init__(self, env, states, actions):
        super(DAC, self).__init__(states, actions)
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.μ, self.Q = Net(), QNet()

    def learn(self):
        # get state $s_t$
        st = self.states.vec_data
        st = torch.tensor(st, requires_grad=True, dtype=torch.float)
        
        # get action $a_t$ with policy μ and $s_t$
        at = self.μ(st)
        
        # copy $a_t$ for gradient
        at2 = at.data
        at2.requires_grad = True
        
        # ε-greedy
        if np.random.uniform() < EPSILON:
            a_env = at.data.numpy()
        else:
            a_env = self.env.action.space.sample()

        # apply action
        self.set_action_data(a_env)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_, rt, done, info = self.env.step(a_env)
        
        # cast to tensor
        st_ = torch.tensor(st_, dtype=torch.float)
        
        # get action $a_{t+1}$ with policy μ and $s_{t+1}$
        at_ = self.μ(st_)
        
        # get action values Q and Q_
        Q = self.Q(st,at2)
        Q_ = self.Q(st_,at_)
        
        # δ_t = r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        δt = rt - 1 + γ * Q_[0] - Q[0]
        
        self.Q.zero_grad()
        Q.backward()
        for w in self.Q.parameters():
            # w_{t+1} = w_t + α_w * δ_t * ∇_w Q(s_t, a_t)
            w.data = w.data + αw * δt * w.grad

        # ∇_a Q(s_t, a_t)|_{a=μ(s)}
        ΔaQ = at2.grad

        self.μ.zero_grad()
        at.backward(ΔaQ)
        for θ in self.μ.parameters():
            # θ_{t+1} = θ_t + α_θ * ∇_θ μ(s_t) ∇_a Q
            θ.data = θ.data + αθ * θ.grad
        return rt

if __name__=="__main__":
    sim = prl.simulators.Bullet(render=False)
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    states = BasePositionState(manipulator) + JointPositionState(manipulator) + JointVelocityState(manipulator) + PositionState(box,world)
    action = JointPositionAction(manipulator)
    reward = TerminalReward(HasPickedAndPlacedCondition(manipulator,box,world))
    env = Env(world, states, rewards=reward,actions=action)
    
    env.reset()
    dac = DAC(env, states, action)
    num_episodes = 1000
    t_episode = 1000
    for i in count():
        if i>=num_episodes:
            break
        print("Episode {}".format(i))
        manipulator.reset_joint_states()
        cnt = 0
        for t in range(t_episode):
            cnt += dac.learn()
        print("Total reward: ", cnt)
