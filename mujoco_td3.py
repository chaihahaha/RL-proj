import gym

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count

NOISE_EPSILON = 0.1
MEMORY_CAPACITY = 5120
EPSILON = 0.01
RAND_EPSILON = 0.3
ACTION_L2 = 0.5
γ = 0.981
LR = 1e-3
BATCH_SIZE = 128
tau = 0.05
t_episode = 50

class μNet(nn.Module):
    def __init__(self, max_action):
        super(μNet, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(N_STATES, 300)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 1e-5)   # initialization
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 1e-5)   # initialization
        self.fc5 = nn.Linear(300, 300)
        self.fc5.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(300, N_ACTIONS)
        self.out.weight.data.normal_(0, 1e-5)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        #x = self.act(self.fc3(x))
        #x = self.act(self.fc4(x))
        #x = self.act(self.fc5(x))
        x = self.out(x)
        x = self.tanh(x) * self.max_action
        return x
        
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES+N_ACTIONS, 300)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(300, 300)
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
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.cat([x1,x2],dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        #x = self.act(self.out1(x))
        #x = self.act(self.out2(x))
        #x = self.act(self.out3(x))
        out = self.out(x)
        return out

class TD3(object):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        low = env.action_space.low
        high = env.action_space.high
        self.max_action = np.max(high)
        self.μ = μNet(self.max_action)
        self.μ.to(device)
        self.Q1, self.Q2 = QNet(), QNet()
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
            self.Q1_tar, self.Q2_tar = QNet(), QNet()
        self.μ_tar.to(device)
        self.Q1_tar.to(self.device)
        self.Q2_tar.to(self.device)
        self.μ_tar.load_state_dict(self.μ.state_dict())
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())
        self.memory = torch.zeros((MEMORY_CAPACITY,t_episode,N_STATES*2+N_ACTIONS+2),device=self.device)
        self.cnt_step = 0
        self.cnt_epi = 0

    def learn(self, st):
        lossμ, lossQ1, lossQ2 = 0,0,0
    
        # get state $s_t$
        st = np.concatenate([st["observation"],st["achieved_goal"],st["desired_goal"]])
        st = torch.tensor(st).float().to(self.device).unsqueeze(0)
        
        # compute action with actor μ
        at = self.μ(st)
        at += EPSILON*torch.randn(at.shape,device=self.device)
        
        # cast to numpy
        if np.random.uniform() < RAND_EPSILON:
            at_np = self.env.action_space.sample()
        else:
            at_np = (at.data.cpu()).numpy()[0]

        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_dic, rt, done, info = self.env.step(at_np)
        st_ = np.concatenate([st_dic["observation"],st_dic["achieved_goal"],st_dic["desired_goal"]])
        st_ = torch.tensor(st_).float().to(self.device).unsqueeze(0)

        # keep (st, at, rt, st_) in memory
        index = self.cnt_epi % MEMORY_CAPACITY
        t_step = self.cnt_step % t_episode
        self.memory[index,t_step,:N_STATES] = st.detach()
        self.memory[index,t_step,N_STATES:N_STATES+N_ACTIONS] = torch.tensor(at_np)
        self.memory[index,t_step,-N_STATES-2] = torch.tensor(rt)
        mask = 0. if done else γ
        self.memory[index,t_step,-N_STATES-1] = mask
        self.memory[index,t_step,-N_STATES:] = st_.detach()
        

        # randomly sample from memory
        right = self.cnt_epi+1 if self.cnt_epi<MEMORY_CAPACITY else MEMORY_CAPACITY
        right_t = self.cnt_step+1 if self.cnt_step<t_episode else t_episode
        random_index = np.random.choice(right, BATCH_SIZE)
        s = self.memory[random_index,:right_t,:N_STATES].flatten(0,1)
        a = self.memory[random_index,:right_t,N_STATES:N_STATES+N_ACTIONS].flatten(0,1)
        r = self.memory[random_index,:right_t,-N_STATES-2:-N_STATES-1].flatten(0,1)
        mask = self.memory[random_index,:right_t,-N_STATES-1:-N_STATES].flatten(0,1)
        s_ = self.memory[random_index,:right_t,-N_STATES:].flatten(0,1)
        
        lq1, lq2 = self.update_critic(s,a,r,mask,s_)
        lossQ1 += lq1
        lossQ2 += lq2
        
        lossμ += self.update_actor(s)
    
        update_pairs = [(self.μ, self.μ_tar), (self.Q1, self.Q1_tar), (self.Q2, self.Q2_tar)]
        for i, i_tar in update_pairs:
            p = i.named_parameters()
            p_tar = i_tar.named_parameters()
            d_tar = dict(p_tar)
            for name, param in p:
                d_tar[name].data = tau*param.data + (1-tau) * d_tar[name].data
        if done:
            # HER replace goal
            recall_epi = self.memory[self.cnt_epi % MEMORY_CAPACITY]

            # replace goal pos with end effector pos
            fake_goal = recall_epi[-1,-6:-3].clone()
            recall_epi[:,N_STATES-3:N_STATES] = fake_goal
            recall_epi[:,-3:] = fake_goal
            for i in range(t_episode):
                recall_epi[i,-N_STATES-2] = torch.tensor(self.env.compute_reward(achieved_goal=recall_epi[i,-6:-3].cpu(), desired_goal=fake_goal.cpu(), info=info))    
            self.cnt_epi += 1
            self.memory[self.cnt_epi % MEMORY_CAPACITY] = recall_epi
            self.cnt_epi += 1
            
        self.cnt_step += 1
        return st_dic, rt, done, info, lossμ, lossQ1, lossQ2
        
    def update_actor(self, si):
        ai = self.μ(si)
        Q1 = self.Q1(si, ai)
        lossμ = -torch.mean(Q1) + ACTION_L2 * torch.mean(ai**2)
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        lossμ.backward()
        self.optimμ.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        return lossμ.item()
        
        
    def update_critic(self,si,ai,ri,mask,si_):
        with torch.no_grad():
            ai_ = self.μ_tar(si_)
            ai_ += NOISE_EPSILON*torch.randn(ai_.shape,device=self.device)
        
            Q1_ = self.Q1_tar(si_,ai_)
            Q2_ = self.Q2_tar(si_,ai_)
        
        y = ri + mask * torch.min(Q1_, Q2_)

        Q1 = self.Q1(si, ai) 
        lossQ1 = torch.mean((y-Q1)**2)
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        lossQ1.backward()
        self.optimQ1.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        
        Q2 = self.Q2(si, ai)
        lossQ2 = torch.mean((y-Q2)**2)
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        lossQ2.backward()
        self.optimQ2.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        return lossQ1.item(),lossQ2.item() 
    
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
    env=gym.make("FetchReach-v1")
    s_sample = [v for k,v in env.observation_space.sample().items()]
    STATES_SHAPE = [v.shape[0] for v in s_sample]
    N_STATES = sum(STATES_SHAPE)
    N_ACTIONS = env.action_space.sample().shape[0]
    td3 = TD3(env, "cuda")
    #td3.load("td3_gym.ckpt")
    num_episodes = 30000
    save_freq = 200
    
    n_samples = 10
    n_success = 0
    s_reward = 0
    sum_lossμ = 0
    sum_lossQ1 = 0
    sum_lossQ2 = 0
    tik = time.time()
    for i in range(1,num_episodes+1):
        s = env.reset()
        # run an episode
        done = False
        while not done:
#            env.render()
            s, reward, done, info, lossμ, lossQ1, lossQ2 = td3.learn(s)
            sum_lossμ += lossμ
            sum_lossQ1 += lossQ1
            sum_lossQ2 += lossQ2
            s_reward += reward
            if done:
                break
        n_success += 1 if info['is_success'] else 0
        #print("SUCCESS" if done else "FAIL",flush=True)
        
        # collect statistics of #n_samples results
        if i%n_samples==0:
            tok = time.time()
            print("Epoch {}\tSuc rate: {:.2f}\tAvg reward: {:.2f}\tLossμ:{:.3f}\tLossQ1:{:.3f}\tLossQ2:{:.3f}\tTime: {:.1f}".format(int(i/n_samples),n_success/n_samples,s_reward/n_samples,sum_lossμ/n_samples, sum_lossQ1/n_samples, sum_lossQ2/n_samples, tok-tik),flush=True)
            sum_lossμ = 0
            sum_lossQ1 = 0
            sum_lossQ2 = 0
            n_success = 0
            s_reward = 0
            tik = time.time()
        if i%save_freq==0:
            td3.save("td3_gym.ckpt")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
