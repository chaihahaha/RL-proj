import gym

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count

NOISE_EPSILON = 0.2
NOISE_CLIP = 0.5
MEMORY_CAPACITY = 5120
RAND_EPSILON = 0.3
ACTION_L2 = 0.0
γ = 0.99
LR = 3e-4
BATCH_SIZE = 8
tau = 0.005
t_episode = 50
policy_freq = 2
start_timesteps = 25e3

class μNet(nn.Module):
    def __init__(self, max_action):
        super(μNet, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.fc3 = nn.Linear(256, 256)
        self.fc3.weight.data.normal_(0, 1e-5)   # initialization
        self.fc4 = nn.Linear(256, 256)
        self.fc4.weight.data.normal_(0, 1e-5)   # initialization
        self.fc5 = nn.Linear(256, 256)
        self.fc5.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(256, N_ACTIONS)
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
        self.fc1 = nn.Linear(N_STATES+N_ACTIONS, 256)
        self.fc1.weight.data.normal_(0, 1e-5)   # initialization
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 1e-5)   # initialization
        self.out1 = nn.Linear(256, 256)
        self.out1.weight.data.normal_(0, 1e-5)   # initialization
        self.out2 = nn.Linear(256, 256)
        self.out2.weight.data.normal_(0, 1e-5)   # initialization
        self.out3 = nn.Linear(256, 256)
        self.out3.weight.data.normal_(0, 1e-5)   # initialization
        self.out = nn.Linear(256, 1)
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
    def learn(self, st):
        lossμ, lossQ1, lossQ2 = 0,0,0
    
        # get state $s_t$
        st = np.concatenate([st["observation"],st["achieved_goal"],st["desired_goal"]])
        st = torch.tensor(st).float().to(self.device).unsqueeze(0)
        
        # cast to numpy
        # compute action with actor μ
        with torch.no_grad():
            at = self.μ(st)
        at = (at).clamp(-self.max_action, self.max_action)
        at_np = at.detach().cpu().numpy()[0]

        # step in environment to get next state $s_{t+1}$, reward $r_t$
        st_dic, rt, done, info = self.env.step(at_np)
        st_ = np.concatenate([st_dic["observation"],st_dic["achieved_goal"],st_dic["desired_goal"]])
        st_ = torch.tensor(st_).float().to(self.device).unsqueeze(0)

        return st_dic, rt, done, info
        
    def load(self, filename):
        state_dicts = torch.load(filename)
        self.μ.load_state_dict(state_dicts['μ'])
        self.Q1.load_state_dict(state_dicts['Q1'])
        self.Q2.load_state_dict(state_dicts['Q2'])

def success(reward):
    return reward >= -0.5

if __name__=="__main__":
    env=gym.make("FetchReach-v1")
    s_sample = [v for k,v in env.observation_space.sample().items()]
    STATES_SHAPE = [v.shape[0] for v in s_sample]
    N_STATES = sum(STATES_SHAPE)
    N_ACTIONS = env.action_space.sample().shape[0]
    print(STATES_SHAPE)
    print(N_ACTIONS)
    td3 = TD3(env, "cuda")
    td3.load("td3_gym.ckpt")
    num_episodes = 30000
    save_freq = 200
    
    n_samples = 10
    n_success = 0
    s_reward = 0
    tik = time.time()
    for i in range(1,num_episodes+1):
        s = env.reset()
        # run an episode
        done = False
        while not done:
            env.render()
            s, reward, done, info = td3.learn(s)
            s_reward += reward
            if done:
                break
        n_success += 1 if info['is_success'] else 0
        #print("SUCCESS" if done else "FAIL",flush=True)
