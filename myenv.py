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
MEMORY_CAPACITY = 5120
EPSILON = 0.4
γ = 0.994
LR = 1e-3
BATCH_SIZE = 512
TRAIN_FREQ = 20
TARGET_REPLACE_ITER = TRAIN_FREQ
tau = 0.9
t_episode = 100

def layer_norm(layer, std=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
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
        self.fc1 = nn.Linear(N_STATES, 200)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(200, 200)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.fc3 = nn.Linear(200, 200)
        self.fc3.weight.data.normal_(0, 1e-5)   # initialization
        self.fc4 = nn.Linear(200, 200)
        self.fc4.weight.data.normal_(0, 1e-5)   # initialization
        self.fc5 = nn.Linear(200, 200)
        self.fc5.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(200, N_ACTIONS)
        self.out.weight.data.normal_(0, 1e-5)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        layer_norm(self.fc1)
        layer_norm(self.fc2)
        layer_norm(self.fc3)
        layer_norm(self.fc4)
        layer_norm(self.fc5)

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
        self.fc1 = nn.Linear(N_STATES, 200)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(N_ACTIONS, 100)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.out1 = nn.Linear(300, 300)
        self.out1.weight.data.normal_(0, 1e-5)   # initialization
        self.out2 = nn.Linear(300, 300)
        self.out2.weight.data.normal_(0, 1e-5)   # initialization
        self.out3 = nn.Linear(300, 300)
        self.out3.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(300, 1)
        self.out.weight.data.normal_(0, 1e-5)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        layer_norm(self.fc1)
        layer_norm(self.fc2)
        layer_norm(self.out1)
        layer_norm(self.out2)
        layer_norm(self.out3)
        layer_norm(self.out)

    def forward(self, x1, x2):
        x1 = self.act(self.fc1(x1))
        x2 = self.act(self.fc2(x2))
        x = torch.cat([x1,x2],dim=1)
        x = self.act(self.out1(x))
        x = self.act(self.out2(x))
        x = self.act(self.out3(x))
        out = self.out(x)
        return out

class TD3(Policy):
    def __init__(self, env, states, actions,robot, device):
        super(TD3, self).__init__(states, actions)
        self.robot = robot
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.device = device
        self.μ = μNet(actions.bounds())
        self.μ.to(device)
        self.Q1, self.Q2 = QNet(), QNet()
        self.Q1.to(self.device)
        self.Q2.to(self.device)
        trainableμ = list(filter(lambda p: p.requires_grad, self.μ.parameters()))
        self.optimμ = torch.optim.Adam(trainableμ, lr=LR, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-5)
        trainableQ1 = list(filter(lambda p: p.requires_grad, self.Q1.parameters()))
        self.optimQ1 = torch.optim.Adam(trainableQ1, lr=LR, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-5)
        trainableQ2 = list(filter(lambda p: p.requires_grad, self.Q2.parameters()))
        self.optimQ2 = torch.optim.Adam(trainableQ2, lr=LR, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-5)
        self.μ_tar = μNet(actions.bounds())
        self.μ_tar.to(device)
        self.Q1_tar, self.Q2_tar = QNet(), QNet()
        self.Q1_tar.to(self.device)
        self.Q2_tar.to(self.device)
        self.μ_tar.load_state_dict(self.μ.state_dict())
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())
        self.memory = torch.zeros((MEMORY_CAPACITY,N_STATES*2+N_ACTIONS+1),device=self.device)
        self.cnt = 0

    def learn(self, timeout):
        # get state $s_t$
        st = self.states.vec_torch_data.float().to(self.device).unsqueeze(0)
        
        # compute action with actor μ
        at = self.μ(st)
        at += EPSILON*torch.randn(at.shape,device=self.device)
        
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
            update_pairs = [(self.μ, self.μ_tar), (self.Q1, self.Q1_tar), (self.Q2, self.Q2_tar)]
            for i, i_tar in update_pairs:
                p = i.named_parameters()
                p_tar = i_tar.named_parameters()
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
        Qm = self.Q1(si, self.μ(si))
        lossμ = -torch.mean(Qm)
        lossμ.backward(retain_graph=True)
        self.optimμ.step()
        self.optimμ.zero_grad()

        ai_ = self.μ_tar(si_).detach()
        ai_ += EPSILON*torch.randn(ai_.shape,device=self.device)
        Q1_ = self.Q1_tar(si_,ai_).detach()
        Q2_ = self.Q2_tar(si_,ai_).detach()
        y = ri + γ * torch.min(Q1_, Q2_)
        
        Q1 = self.Q1(si, ai)
        lossQ1 = torch.mean((y-Q1)**2)
        lossQ1.backward()
        self.optimQ1.step()
        self.optimQ1.zero_grad()
        
        Q2 = self.Q2(si, ai)
        lossQ2 = torch.mean((y-Q2)**2)
        lossQ2.backward()
        self.optimQ2.step()
        self.optimQ2.zero_grad()

            
    
    def save(self, filename):
        torch.save({'μ':self.μ_tar.state_dict(),
                    'Q1':self.Q1_tar.state_dict(),
                    'Q2':self.Q2_tar.state_dict()}, filename)
    
    def load(self, filename):
        state_dicts = torch.load(filename)
        self.μ.load_state_dict(state_dicts['μ'])
        self.Q1.load_state_dict(state_dicts['Q1'])
        self.Q2.load_state_dict(state_dicts['Q2'])
        self.μ_tar.load_state_dict(state_dicts['μ'])
        self.Q1_tar.load_state_dict(state_dicts['Q1'])
        self.Q2_tar.load_state_dict(state_dicts['Q2'])

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
    action = JointPositionAction(manipulator, kp=manipulator.kp, kd=manipulator.kd)
    r_cond = HasTouchedCondition(manipulator,box,world,0.5)
    t_cond = HasTouchedCondition(manipulator,box,world,0.5)
    reward = TerminalReward(r_cond,subreward=-1,final_reward=0)
    #reward = ApproachBoxReward(manipulator,box,world)
    env = Env(world, states, rewards=reward,actions=action,terminal_conditions=t_cond)
    
    env.reset()
    td3 = TD3(env, states, action,manipulator, "cuda")
    #td3.load("td3.ckpt")
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
            reward, done = td3.learn(timeout)
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
            td3.save("td3.ckpt")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
