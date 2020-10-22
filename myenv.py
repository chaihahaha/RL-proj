import pyrobolearn as prl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.rewards.terminal_rewards import TerminalReward
from pyrobolearn.terminal_conditions import LinkPositionCondition, TerminalCondition
from pyrobolearn.states.body_states import PositionState, VelocityState
from pyrobolearn.states import BasePositionState, JointPositionState, JointVelocityState
from pyrobolearn.actions.robot_actions.joint_actions import JointPositionAction
from pyrobolearn.tasks.reinforcement import RLTask


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

class HasPickedAndLiftedCondition(TerminalCondition):
    def __init__(self, robot, box, world):
        self.robot = robot
        self.box = box
        self.world = world
    def check(self):
        x,y,z = self.world.get_body_position(self.box)
        return z>0.5
        
class HasTouchedCondition(TerminalCondition):
    def __init__(self, robot, box, world, radius):
        self.robot = robot
        self.box = box
        self.world = world
        self.radius = radius
    def check(self):
        end_effector = self.robot.get_end_effector_ids()[-1]
        x1,y1,z1 = self.robot.get_link_world_positions(end_effector)
        x2,y2,z2 = self.world.get_body_position(box)
        dx,dy,dz = x1-x2,y1-y2,z1-z2
        return dx**2 + dy**2 + dz**2 <self.radius**2

class μNet(nn.Module):
    def __init__(self, bounds):
        super(μNet, self).__init__()
        self.bounds = bounds
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.out(x)
        actions = []
        for a in range(N_ACTIONS):
            actions.append(self.clip(x[a],self.bounds[a,0],self.bounds[a,1]))
        return torch.stack(actions,axis=0).view(N_ACTIONS)
        
    def clip(self, x, x_min, x_max):
        return x_min + (x_max-x_min)*(self.tanh(x)+1)/2
        
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_ACTIONS, 50)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out1 = nn.Linear(100, 100)
        self.out1.weight.data.normal_(0, 0.1)   # initialization
        self.out2 = nn.Linear(100, 1)
        self.out2.weight.data.normal_(0, 0.1)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2):
        x1 = self.act(self.fc1(x1))
        x2 = self.act(self.fc2(x2))
        x = self.act(torch.stack([x1,x2],1).view(-1,100))
        x = self.act(self.out1(x))
        x = self.act(self.out2(x))
        return x

# Deterministic Actor Critic
class DDPG_AC(Policy):
    def __init__(self, env, states, actions, device):
        super(DDPG_AC, self).__init__(states, actions)
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.device = device
        self.μ, self.Q = μNet(bounds), QNet()
        self.μ.to(device)
        self.Q.to(device)

    def learn(self):
        # get state $s_t$
        st = self.states.vec_data
        st = torch.tensor(st, requires_grad=True, dtype=torch.float,device=self.device)
        
        # get action $a_t$ with policy μ and $s_t$
        at = self.μ(st)
        
        # copy $a_t$ for gradient
        at2 = at.data
        at2.requires_grad = True
        
        # ε-greedy
        if np.random.uniform() < EPSILON:
            a_env = at.data.cpu().numpy()
        else:
            a_env = self.env.action.space.sample()

        # apply action
        self.set_action_data(a_env)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_, rt, done, info = self.env.step(a_env)
        manipulator.get_end_effector_ids()[-1]
        # cast to tensor
        st_ = torch.tensor(st_, dtype=torch.float,device=self.device)
        
        # get action $a_{t+1}$ with policy μ and $s_{t+1}$
        at_ = self.μ(st_)
        
        # get action values Q and Q_
        Q = self.Q(st,at2)
        Q_ = self.Q(st_,at_)
        #print(Q,Q_)
        
        # δ_t = r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        δt = rt + γ * Q_[0] - Q[0]
        
        self.Q.zero_grad()
        Q.backward()
        for w in self.Q.parameters():
            # w_{t+1} = w_t + α_w * δ_t * ∇_w Q(s_t, a_t)
            w.data = w.data + αw * δt * w.grad
        #print(sum([torch.sum(i) for i in self.μ.parameters()]))
        # ∇_a Q(s_t, a_t)|_{a=μ(s)}
        ΔaQ = at2.grad

        self.μ.zero_grad()
        at.backward(ΔaQ)
        for θ in self.μ.parameters():
            # θ_{t+1} = θ_t + α_θ * ∇_θ μ(s_t) ∇_a Q
            θ.data = θ.data + αθ * θ.grad
        return rt, done

def success(reward):
    return reward >= 0

if __name__=="__main__":
    sim = prl.simulators.Bullet(render=True)
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0.2),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    states = BasePositionState(manipulator) + JointPositionState(manipulator) + JointVelocityState(manipulator) + PositionState(box,world)
    action = JointPositionAction(manipulator)
    bounds = action.bounds()
    r_cond = HasTouchedCondition(manipulator,box,world,0.5)
    t_cond = HasTouchedCondition(manipulator,box,world,0.45)
    reward = TerminalReward(r_cond,subreward=-1,final_reward=0)
    env = Env(world, states, rewards=reward,actions=action,terminal_conditions=t_cond)
    
    env.reset()
    ddpg = DDPG_AC(env, states, action, "cuda")
    num_episodes = 1000
    t_episode = 200
    n_success = 0
    for i in range(num_episodes):
        manipulator.reset_joint_states(np.random.uniform(-1,1,(15)))
        world.move_object(box,[np.random.uniform(-1,1),np.random.uniform(-1,1),0.2])
        for t in range(t_episode):
            reward, done = ddpg.learn()
            if done:
                break
        n_success += 1 if success(reward) else 0
        print("Episode {}\tSuccess rate: {:.4f}\t".format(i,n_success/(i+1)), "SUCCESS" if success(reward) else "FAIL")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
