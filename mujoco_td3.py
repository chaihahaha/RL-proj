import gym

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count


MEMORY_CAPACITY = 5120
EPSILON = 0.4
γ = 0.994
LR = 1e-3
BATCH_SIZE = 512
TRAIN_FREQ = 20
TARGET_REPLACE_ITER = TRAIN_FREQ
tau = 0.9
t_episode = 50

def layer_norm(layer, std=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
def list_dic2array(s):
    # list of dict to array
    s = [np.concatenate([i for i in j.values()]) for j in s if j!=None]
    s = np.array(s)
    return s
class μNet(nn.Module):
    def __init__(self, low, high):
        super(μNet, self).__init__()
        self.low = low
        self.high = high
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
            x[:,i] = self.clip(x[:,i], self.low[i],self.high[i])
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

class TD3(object):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        low = env.action_space.low
        high = env.action_space.high
        self.μ = μNet(low,high)
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
        self.μ_tar = μNet(low,high)
        self.μ_tar.to(device)
        self.Q1_tar, self.Q2_tar = QNet(), QNet()
        self.Q1_tar.to(self.device)
        self.Q2_tar.to(self.device)
        self.μ_tar.load_state_dict(self.μ.state_dict())
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())
        self.memory = {"s":[None]*MEMORY_CAPACITY,"a":[np.zeros(N_ACTIONS) for i in range(MEMORY_CAPACITY)],"r":[0.]*MEMORY_CAPACITY,"s_":[None]*MEMORY_CAPACITY}
        self.cnt = 0

    def learn(self, st_dic):
        # get state $s_t$
        st = np.concatenate([i for i in st_dic.values()])
        st = torch.tensor(st).float().to(self.device).unsqueeze(0)
        
        # compute action with actor μ
        at = self.μ(st)
        at += EPSILON*torch.randn(at.shape,device=self.device)
        
        # cast to numpy
        at_np = (at.data.cpu()).numpy()[0]

        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_, rt, done, info = self.env.step(at_np)

        # keep (st, at, rt, st_) in memory
        index = self.cnt % MEMORY_CAPACITY
        self.memory["s"][index] = st_dic
        self.memory["a"][index] = at_np
        self.memory["r"][index] = rt
        self.memory["s_"][index] = st_
        
        if (self.cnt + 1) % TRAIN_FREQ == 0:
            # randomly sample from memory
            right = self.cnt+1 if self.cnt<MEMORY_CAPACITY else MEMORY_CAPACITY
            random_index = np.random.choice(right, BATCH_SIZE)
            
            s = list_dic2array(self.memory["s"])[random_index]
            a = np.array(self.memory["a"])[random_index]
            r = np.array(self.memory["r"])[random_index]
            s_ = list_dic2array(self.memory["s_"])[random_index]
            
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
        if done:
            right = self.cnt % MEMORY_CAPACITY 
            recall_epi = {"s":[None]*t_episode,"a":[None]*t_episode,"r":[None]*t_episode,"s_":[None]*t_episode}
            
            for k in ["s","a","r","s_"]:
                for i in range(t_episode):
                    value = self.memory[k][(right-i) % MEMORY_CAPACITY]
                    recall_epi[k][i] = value if isinstance(value,float) else value.copy()

            # replace goal pos with end effector pos
            fake_goal = recall_epi["s"][0]["achieved_goal"].copy()
            for k in ["s","s_"]:
                for i in range(t_episode):
                    recall_epi[k][i]["desired_goal"] = fake_goal
            recall_epi["r"][0] = 0.
            for i in range(t_episode):
                self.cnt += 1
                for k in ["s","a","r","s_"]:
                    value = recall_epi[k][i]
                    self.memory[k][self.cnt % MEMORY_CAPACITY] = value if isinstance(value,float) else value.copy()
            
        self.cnt += 1
        return st_, rt, done
        
    def train(self, si, ai, ri, si_):
        si = torch.tensor(si).float().to(self.device)
        ai = torch.tensor(ai).float().to(self.device)
        ri = torch.tensor(ri).float().to(self.device).unsqueeze(1)
        si_ = torch.tensor(si).float().to(self.device)
        
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
    env=gym.make("FetchReach-v1")
    s_sample = [v for k,v in env.observation_space.sample().items()]
    STATES_SHAPE = [v.shape[0] for v in s_sample]
    N_STATES = sum(STATES_SHAPE)
    N_ACTIONS = env.action_space.sample().shape[0]
    td3 = TD3(env, "cuda")
    #td3.load("td3_gym.ckpt")
    num_episodes = 30000
    save_freq = 200
    
    n_samples = 100
    n_success = 0
    s_reward = 0
    tik = time.time()
    for i in range(1,num_episodes+1):
        s = env.reset()
        # run an episode
        done = False
        while not done:
            env.render()
            s, reward, done = td3.learn(s)
            if done:
                break
        n_success += 1 if (reward==0.) else 0
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
            td3.save("td3_gym.ckpt")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
