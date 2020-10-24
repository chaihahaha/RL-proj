import pyrobolearn as prl
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.rewards.terminal_rewards import TerminalReward
from pyrobolearn.terminal_conditions import LinkPositionCondition, TerminalCondition
from pyrobolearn.states.body_states import DistanceState, PositionState, VelocityState
from pyrobolearn.states import LinkPositionState, JointPositionState, JointVelocityState, LinkWorldPositionState, SensorState
from pyrobolearn.actions.robot_actions.joint_actions import JointPositionAction
from pyrobolearn.tasks.reinforcement import RLTask
from pyrobolearn.rewards.reward import Reward
from pyrobolearn.robots.sensors import RGBCameraSensor

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count

N_ACTIONS = 15
MEMORY_CAPACITY = 1000
EPSILON = 0.1
γ = 0.9
LR = 1e-3
BATCH_SIZE = 512
TRAIN_FREQ = 20
TARGET_REPLACE_ITER = TRAIN_FREQ
tau = 0.9
t_episode = 100

class MyCameraState(SensorState):
    def __init__(self, camera):
        self._sensor = camera
        super(MyCameraState, self).__init__(camera)
    def _read(self):
        if self._update:
            self.sensor.sense(apply_noise=True)

        # get the data from the sensor
        self.data = self.sensor.data.reshape(-1)
    def _reset(self):
        self._cnt = 0
        self._update = False
        self._sensor.enable()
        while self._sensor.sense() is None:
            self._sensor.sense()
        self._read()

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
        x2,y2,z2 = self.world.get_body_position(self.box)
        dx,dy,dz = x1-x2,y1-y2,z1-z2
        return dx**2 + dy**2 + dz**2 <=self.radius**2
        
class ApproachBoxReward(Reward):
    def __init__(self, robot, box, world, range=(-np.infty,0)):
        self.robot = robot
        self.box = box
        self.world = world
        self.range = range
        
    def _compute(self):
        """Compute the reward."""
        end_effector = self.robot.get_end_effector_ids()[-1]
        x1,y1,z1 = self.robot.get_link_world_positions(end_effector)
        x2,y2,z2 = self.world.get_body_position(self.box)
        dx,dy,dz = x1-x2,y1-y2,z1-z2
        self.value = -(dx**2+dy**2+dz**2)**0.5
        return self.value
        
class μNet(nn.Module):
    def __init__(self, bounds):
        super(μNet, self).__init__()
        self.bounds = bounds
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.fc3 = nn.Linear(100, 50)
        self.fc3.weight.data.normal_(0, 0.1)   # initialization
        self.fc4 = nn.Linear(50, 200)
        self.fc4.weight.data.normal_(0, 0.1)   # initialization
        self.fc5 = nn.Linear(200, 50)
        self.fc5.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.out(x)
        for i in range(N_ACTIONS):
            x[:,i] = self.clip(x[:,i], self.bounds[i,0],self.bounds[i,1])
        return x
        
    def clip(self, x, x_min, x_max):
        return x_min + (x_max-x_min)*(self.tanh(x)+1)/2
        
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_ACTIONS, 50)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.out1 = nn.Linear(100, 50)
        self.out1.weight.data.normal_(0, 0.1)   # initialization
        self.out2 = nn.Linear(50, 100)
        self.out2.weight.data.normal_(0, 0.1)   # initialization
        self.out3 = nn.Linear(100, 50)
        self.out3.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2):
        x1 = self.act(self.fc1(x1))
        x2 = self.act(self.fc2(x2))
        x = torch.cat([x1,x2],dim=1)
        x = self.act(self.out1(x))
        x = self.act(self.out2(x))
        x = self.act(self.out3(x))
        out = self.out(x)
        return out

class ACNet(nn.Module):
    def __init__(self, bounds):
        super(ACNet, self).__init__()
        self.μ = μNet(bounds)
        self.Q = QNet()
    
    def forward(self, s):
        self.a = self.μ(s)
        q = self.Q(s,self.a)
        return q

# Deterministic Actor Critic
class DDPG_AC(Policy):
    def __init__(self, env, states, actions,robot, device):
        super(DDPG_AC, self).__init__(states, actions)
        self.robot = robot
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.device = device
        self.ac = ACNet(actions.bounds())
        self.ac.to(device)
        trainableμ = list(filter(lambda p: p.requires_grad, self.ac.μ.parameters()))
        self.optimμ = torch.optim.Adam(trainableμ, lr=LR, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-5)
        trainableQ = list(filter(lambda p: p.requires_grad, self.ac.Q.parameters()))
        self.optimQ = torch.optim.Adam(trainableQ, lr=LR, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-5)
        self.ac_tar = ACNet(actions.bounds())
        self.ac_tar.to(device)
        
        self.memory = torch.zeros((MEMORY_CAPACITY,N_STATES*2+N_ACTIONS+1),device=self.device)
        self.cnt = 0

    def learn(self, timeout):
        # get state $s_t$
        st = self.states.vec_torch_data.float().to(self.device).unsqueeze(0)
        
        # compute action with actor μ
        at = self.ac.μ(st) + EPSILON*torch.randn((st.shape[0],N_ACTIONS),device=self.device)
        
        # cast to numpy
        at_np = (at.data.cpu()).numpy()[0]

        # apply action
        self.set_action_data(at_np)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_, rt, done, info = self.env.step()

        # cast to tensor
        st_ = torch.tensor(st_, dtype=torch.float,device=self.device)
        
        # keep (st, at, rt, st_) in memory
        index = self.cnt % MEMORY_CAPACITY
        self.memory[index,:N_STATES] = st.detach()
        self.memory[index,N_STATES:N_STATES+N_ACTIONS] = at.detach()
        self.memory[index,-N_STATES-1] = torch.tensor(rt)
        self.memory[index,-N_STATES:] = st_.detach()
        
        if (self.cnt + 1) % TRAIN_FREQ == 0:
            # randomly sample from memory
            right = self.cnt+1 if self.cnt<MEMORY_CAPACITY else MEMORY_CAPACITY
            mem = self.memory[:right,:]
            random_index = np.random.choice(right, BATCH_SIZE)
            s = self.memory[random_index,:N_STATES]
            a = self.memory[random_index,N_STATES:N_STATES+N_ACTIONS]
            r = self.memory[random_index,-N_STATES-1:-N_STATES]
            s_ = self.memory[random_index,-N_STATES:]
            
            self.train(s,a,r,s_)
        
        # update target network
        if  (self.cnt+1) % TARGET_REPLACE_ITER == 0:
            # update target network
            p = self.ac.named_parameters()
            p_tar = self.ac_tar.named_parameters()
            d_tar = dict(p_tar)
            for name, param in p:
                d_tar[name].data = tau*param.data + (1-tau) * d_tar[name].data
        
        # HER replace goal
        if timeout and (not done):
            right = self.cnt % MEMORY_CAPACITY 
            recall_epi = torch.zeros((t_episode, self.memory.shape[1]))
            mem = self.memory.detach().clone()
            
            for i in range(t_episode):
                recall_epi[i,:] = mem[(right-i) % MEMORY_CAPACITY,:]
                
            # replace goal pos with end effector pos
            s = recall_epi[:,:N_STATES]
            a = recall_epi[:,N_STATES:N_STATES+N_ACTIONS]
            r = recall_epi[:,-N_STATES-1:-N_STATES]
            s_ = recall_epi[:,-N_STATES:]
            
            end_effector = self.robot.get_end_effector_ids()[-1]
            x1,y1,z1 = self.robot.get_link_world_positions(end_effector)
            fake_goal = torch.tensor([[x1,y1,z1]])
            
            s[:,-STATES_SHAPE[-1]:] = fake_goal
            s_[:,-STATES_SHAPE[-1]:] = fake_goal
            r[0,0] = 0.
            recall_epi = torch.cat([s,a,r,s_],1)
            for i in range(t_episode):
                self.cnt += 1
                self.memory[self.cnt % MEMORY_CAPACITY] = recall_epi[i].detach()
            
        self.cnt += 1
        return rt, done
        
    def train(self, si, ai, ri, si_):
        Qm = self.ac(si)
        lossμ = -torch.mean(Qm)
        lossμ.backward(retain_graph=True)
        self.optimμ.step()
        self.optimμ.zero_grad()

        Q = self.ac.Q(si,ai)
        Q_ = self.ac_tar(si_).detach()
        
        # δ_t = r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        δt = ri + γ * Q_ - Q
        lossQ = torch.mean(δt**2)
        lossQ.backward()
        self.optimQ.step()
        self.optimQ.zero_grad()

            
    
    def save(self, filename):
        torch.save({'net':self.ac.state_dict()}, filename)
    
    def load(self, filename):
        state_dicts = torch.load(filename)
        self.ac.load_state_dict(state_dicts['net'])

def success(reward):
    return reward >= -0.5

if __name__=="__main__":
    sim = prl.simulators.Bullet(render=False)
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0.2),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    end_effector = manipulator.get_end_effector_ids()[-1]
    camera = RGBCameraSensor(sim, manipulator, end_effector, 16,16)
    states = MyCameraState(camera) + LinkWorldPositionState(manipulator) + JointPositionState(manipulator) + JointVelocityState(manipulator) + PositionState(box,world)
    STATES_SHAPE = [i.shape[0] for i in states()]
    print(STATES_SHAPE)
    N_STATES = sum(STATES_SHAPE)
    action = JointPositionAction(manipulator)
    r_cond = HasTouchedCondition(manipulator,box,world,0.5)
    t_cond = HasTouchedCondition(manipulator,box,world,0.5)
    reward = TerminalReward(r_cond,subreward=-1,final_reward=0)
    #reward = ApproachBoxReward(manipulator,box,world)
    env = Env(world, states, rewards=reward,actions=action,terminal_conditions=t_cond)
    
    env.reset()
    ddpg = DDPG_AC(env, states, action,manipulator, "cuda")
    #ddpg.load("ddpg.ckpt")
    num_episodes = 30000
    save_freq = 200
    
    n_samples = 100
    n_success = 0
    s_reward = 0
    tik = time.time()
    for i in range(1,num_episodes+1):
        # randomly reset robot joint and box position
        manipulator.reset_joint_states(np.random.uniform(-1,1,(15)))
        world.move_object(box,[np.random.uniform(-1,1),np.random.uniform(-1,1),0.2])
        
        # run an episode
        for t in range(t_episode):
            timeout = t==t_episode-1
            reward, done = ddpg.learn(timeout)
            if done:
                break
        n_success += 1 if done else 0
        s_reward += reward
        #print("SUCCESS" if done else "FAIL",flush=True)
        
        # collect statistics of #n_samples results
        if i%n_samples==0:
            tok = time.time()
            print("{}\tSuc rate: {:.2f}\tAvg reward: {:.2f}\tTime: {:.1f}".format(int(i/n_samples),n_success/n_samples,s_reward/n_samples,tok-tik),flush=True)
            n_success = 0
            s_reward = 0
            tik = time.time()
        if i%save_freq==0:
            ddpg.save("ddpg.ckpt")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
