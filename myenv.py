import pyrobolearn as prl
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.states.body_states import DistanceState, PositionState, VelocityState
from pyrobolearn.states import JointPositionState, JointVelocityState, LinkWorldPositionState, LinkWorldVelocityState
from pyrobolearn.actions import JointPositionChangeAction

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count

N_ACTIONS = 15
MEMORY_CAPACITY = int(1e5)
NOISE_EPSILON = 0.1
RAND_EPSILON = 0.3
ACTION_L2 = 0.5
γ = 0.991
LR = 1e-3
BATCH_SIZE = 512
TRAIN_FREQ = 2
TARGET_REPLACE_ITER = TRAIN_FREQ * 2
polyak = 0.95
t_episode = 100

def picked_and_lifted_reward(box_pos):
    assert len(box_pos) == 3
    x,y,z = box_pos
    return z>0.5
        
def touched_reward(end_effector_pos, box_pos):
    assert len(end_effector_pos) == 3
    assert len(box_pos) == 3
    x1,y1,z1 = end_effector_pos
    x2,y2,z2 = box_pos
    dx,dy,dz = x1-x2,y1-y2,z1-z2
    return 0. if dx**2 + dy**2 + dz**2 <=0.5**2 else -1.
        
def approach_box_reward(end_effector_pos, box_pos):
    assert len(end_effector_pos) == 3
    assert len(box_pos) == 3
    x1,y1,z1 = end_effector_pos
    x2,y2,z2 = box_pos
    dx,dy,dz = x1-x2,y1-y2,z1-z2
    value = -(dx**2+dy**2+dz**2)**0.5
    return value
        
class μNet(nn.Module):
    def __init__(self, max_action):
        super(μNet, self).__init__()
        self.max_action = max_action
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
        self.norm = nn.LayerNorm(200, elementwise_affine=False)
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.out(x)
        x = self.tanh(x) * self.max_action
        return x
        
class QNet(nn.Module):
    def __init__(self, max_action):
        super(QNet, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 300)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 1e-5)   # initialization
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 1e-5)   # initialization
        self.fc5 = nn.Linear(300, 300)
        self.fc5.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(300, 1)
        self.out.weight.data.normal_(0, 1e-5)   # initialization
        self.norm = nn.LayerNorm(300, elementwise_affine=False)
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2):
        x = torch.cat([x1,x2/self.max_action],dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        out = self.out(x)
        return out

class TD3(Policy):
    def __init__(self, env, states, actions, robot, reward, device):
        super(TD3, self).__init__(states, actions)
        self.robot = robot
        self.reward = reward
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.device = device
        self.max_action = np.max(np.abs(actions.bounds()))
        self.μ = μNet(self.max_action)
        self.μ.to(device)
        self.Q1, self.Q2 = QNet(self.max_action), QNet(self.max_action)
        self.Q1.to(self.device)
        self.Q2.to(self.device)
        trainableμ = list(filter(lambda p: p.requires_grad, self.μ.parameters()))
        self.optimμ = torch.optim.Adam(trainableμ, lr=LR)
        trainableQ1 = list(filter(lambda p: p.requires_grad, self.Q1.parameters()))
        self.optimQ1 = torch.optim.Adam(trainableQ1, lr=LR)
        trainableQ2 = list(filter(lambda p: p.requires_grad, self.Q2.parameters()))
        self.optimQ2 = torch.optim.Adam(trainableQ2, lr=LR)
        with torch.no_grad():
            self.μ_tar = μNet(self.max_action)
            self.Q1_tar, self.Q2_tar = QNet(self.max_action), QNet(self.max_action)
        self.μ_tar.to(device)
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
        at = self.μ(st).detach()
        at += NOISE_EPSILON*torch.randn(at.shape,device=self.device)
        #print(self.Q1(st,at).cpu()[0,0])
        
        # cast to numpy
        if np.random.uniform() < RAND_EPSILON:
            at_np = self.env.action.space.sample()
        else:
            at_np = (at.data.cpu()).numpy()[0]

        # apply action
        self.set_action_data(at_np)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_, _, done, info = self.env.step()
        # take first state as achieved goal, take last state as desired goal
        rt = self.reward(self.states()[0],self.states()[-1])

        # cast to tensor
        st_ = torch.tensor(st_, dtype=torch.float,device=self.device)
        
        # keep (st, at, rt, st_) in memory
        index = self.cnt % MEMORY_CAPACITY
        self.memory[index,:N_STATES] = st.detach()
        self.memory[index,N_STATES:N_STATES+N_ACTIONS] = torch.tensor(at_np)
        self.memory[index,-N_STATES-1] = torch.tensor(rt)
        self.memory[index,-N_STATES:] = st_.detach()
        
        if (self.cnt + 1) % TRAIN_FREQ == 0:
            # randomly sample from memory
            right = self.cnt+1 if self.cnt<MEMORY_CAPACITY else MEMORY_CAPACITY
            random_index = np.random.choice(right, BATCH_SIZE)
            s = self.memory[random_index,:N_STATES]
            a = self.memory[random_index,N_STATES:N_STATES+N_ACTIONS]
            r = self.memory[random_index,-N_STATES-1:-N_STATES]
            s_ = self.memory[random_index,-N_STATES:]
            
            self.update_critic(s,a,r,s_)
        
        # update target network
        if  (self.cnt+1) % TARGET_REPLACE_ITER == 0:
            # update target network
            self.update_actor(s)
            update_pairs = [(self.μ, self.μ_tar), (self.Q1, self.Q1_tar), (self.Q2, self.Q2_tar)]
            for i, i_tar in update_pairs:
                p = i.named_parameters()
                p_tar = i_tar.named_parameters()
                d_tar = dict(p_tar)
                for name, param in p:
                    d_tar[name].data = polyak*param.data + (1-polyak) * d_tar[name].data
        
        # HER replace goal
        if timeout:
            right = self.cnt % MEMORY_CAPACITY 
            recall_epi = torch.zeros((t_episode, self.memory.shape[1]))
            for i in range(t_episode):
                recall_epi[i,:] = self.memory[(right-i) % MEMORY_CAPACITY,:]
                
            # replace goal pos with end effector pos
            s = recall_epi[:,:N_STATES]
            a = recall_epi[:,N_STATES:N_STATES+N_ACTIONS]
            r = torch.zeros((t_episode, 1))
            s_ = recall_epi[:,-N_STATES:]
            
            fake_goal = torch.tensor(self.states()[0])
            # make z pos of fake goal real
            fake_goal[-1] = 0.05
            assert fake_goal.shape==torch.Size([3,])
            
            s[:,-STATES_SHAPE[-1]:] = fake_goal
            s_[:,-STATES_SHAPE[-1]:] = fake_goal
            for i in range(t_episode):
                r[i, 0] = self.reward(s_[i, :STATES_SHAPE[0]], fake_goal)
            recall_epi = torch.cat([s,a,r,s_],1)
            for i in range(t_episode):
                self.cnt += 1
                self.memory[self.cnt % MEMORY_CAPACITY] = recall_epi[i].detach()
            
        self.cnt += 1
        return rt, done
        
    def update_actor(self, si):
        ai = self.μ(si)
        Q1 = self.Q1(si, ai)
        lossμ = -torch.mean(Q1) + ACTION_L2 * torch.mean(ai**2)
        lossμ.backward(retain_graph=True)
        self.optimμ.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        
        assert si.shape == torch.Size([BATCH_SIZE, N_STATES])
        assert ai.shape == torch.Size([BATCH_SIZE, N_ACTIONS])
        assert Q1.shape == torch.Size([BATCH_SIZE, 1])
        assert lossμ.shape == torch.Size([])
        
    def update_critic(self,si,ai,ri,si_):
        ai_ = self.μ_tar(si_).detach()
        ai_ += NOISE_EPSILON*torch.randn(ai_.shape,device=self.device)
        
        Q1_ = self.Q1_tar(si_,ai_).detach()
        Q2_ = self.Q2_tar(si_,ai_).detach()
        
        y = ri + γ * torch.min(Q1_, Q2_)

        Q1 = self.Q1(si, ai) 
        lossQ1 = torch.mean((y-Q1)**2)
        lossQ1.backward()
        self.optimQ1.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        
        Q2 = self.Q2(si, ai)
        lossQ2 = torch.mean((y-Q2)**2)
        lossQ2.backward()
        self.optimQ2.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        
        assert si.shape == torch.Size([BATCH_SIZE, N_STATES])
        assert ai.shape == torch.Size([BATCH_SIZE, N_ACTIONS])
        assert ri.shape == torch.Size([BATCH_SIZE, 1])
        assert si_.shape == torch.Size([BATCH_SIZE, N_STATES])
        assert ai_.shape == torch.Size([BATCH_SIZE, N_ACTIONS])
        assert Q1_.shape == torch.Size([BATCH_SIZE, 1])
        assert Q2_.shape == torch.Size([BATCH_SIZE, 1])
        assert y.shape == torch.Size([BATCH_SIZE, 1])
        assert Q1.shape == torch.Size([BATCH_SIZE, 1])
        assert lossQ1.shape == torch.Size([])
        assert Q2.shape == torch.Size([BATCH_SIZE, 1])
        assert lossQ2.shape == torch.Size([])

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
    other_links = manipulator.get_end_effector_ids()[:-1] + manipulator.get_link_ids()
    states = LinkWorldPositionState(manipulator, link_ids=end_effector) + LinkWorldVelocityState(manipulator, link_ids=end_effector) + LinkWorldPositionState(manipulator, link_ids=other_links) + PositionState(box,world)
    STATES_SHAPE = [i.shape[0] for i in states()]
    print(STATES_SHAPE)
    N_STATES = sum(STATES_SHAPE)
    action = JointPositionChangeAction(manipulator)
    env = Env(world, states,actions=action)
    
    env.reset()
    td3 = TD3(env, states, action,manipulator,touched_reward, "cuda")
    
    #td3.load("td3.ckpt")
    num_episodes = 30000
    save_freq = 200
    
    n_samples = 50
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
            s_reward += reward
        n_success += 1 if success(reward) else 0
        #print("SUCCESS" if done else "FAIL",flush=True)
        
        # collect statistics of #n_samples results
        if i%n_samples==0:
            tok = time.time()
            print("Epoch {}\tSuc rate: {:.2f}\tAvg reward: {:.2f}\tTime: {:.1f}".format(int(i/n_samples),n_success/n_samples,s_reward/n_samples,tok-tik),flush=True)
            n_success = 0
            s_reward = 0
            tik = time.time()
        if i%save_freq==0:
            td3.save("td3.ckpt")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
