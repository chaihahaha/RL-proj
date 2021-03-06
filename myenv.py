import pyrobolearn as prl
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.states.body_states import DistanceState, PositionState, VelocityState
from pyrobolearn.states import JointPositionState, JointVelocityState, LinkWorldPositionState, LinkWorldVelocityState
from pyrobolearn.actions import JointPositionChangeAction
from pyrobolearn.terminal_conditions import TerminalCondition

import torch
import torch.nn as nn

import numpy as np
import time
from itertools import count
import pickle

BUFFER_SIZE = int(3e2) # if use intrinsic reward, less memory will encourage forgetting false rt_in
RAND_EPSILON = 0.3
NOISE_CLIP = 0.5
#NOISE_SCALE = 0.2
ACTION_L2 = 1e-3
LR = 1e-3
BATCH_SIZE = 10000
N_BATCHES = 1
TARGET_REPLACE_STEPS = 200
DELAY_ACTOR_STEPS = 400
DELAY_CRITIC_STEPS = 200
polyak = 0.23
t_episode = 200
γ = 0.998
start_timesteps = 1e5
REPLAY_K = 4
LSH_K = 18
β = 2e-2
p_ratio = 2e-4
sim_ratio = 10

def picked_and_lifted_reward(box_pos):
    assert len(box_pos) == 3
    x,y,z = box_pos
    return z>0.5
        
def touched_reward(end_effector_pos, box_pos):
    x1,y1,z1 = end_effector_pos[:,0:1],end_effector_pos[:,1:2],end_effector_pos[:,2:3]
    x2,y2,z2 = box_pos[:,0:1],box_pos[:,1:2],box_pos[:,2:3]
    dx,dy,dz = x1-x2,y1-y2,z1-z2
    return (dx**2 + dy**2 + dz**2 <=0.2**2).float() - 1.
        
def approach_box_reward(end_effector_pos, box_pos):
    assert len(end_effector_pos) == 3
    assert len(box_pos) == 3
    x1,y1,z1 = end_effector_pos
    x2,y2,z2 = box_pos
    dx,dy,dz = x1-x2,y1-y2,z1-z2
    value = -(dx**2+dy**2+dz**2)**0.5
    return value

def intrinsic_counting_reward(state, action):
    freq = hashtable.freq(state, action)
    return (freq + 1)**(-0.5)
        
class ReplayBuffer(object):
    def __init__(self, size, obs_dim, ag_dim, dg_dim, a_dim, r_dim, mask_dim, t_episode, replay_k, reward, meta,device,odevice):
        self.device = device
        self.odevice = odevice
        self.reward = reward
        self.size = size
        self.obs_dim, self.ag_dim, self.dg_dim, self.a_dim, self.r_dim, self.mask_dim = obs_dim, ag_dim, dg_dim, a_dim, r_dim, mask_dim
        self.t_episode = t_episode
        self.sobs = torch.zeros((size,t_episode, obs_dim), device=device)
        self.sag = torch.zeros((size,t_episode, ag_dim), device=device)
        self.sdg = torch.zeros((size,t_episode, dg_dim), device=device)
        
        self.s_obs = torch.zeros((size,t_episode, obs_dim), device=device)
        self.s_ag = torch.zeros((size,t_episode, ag_dim), device=device)
        self.s_dg = torch.zeros((size,t_episode, dg_dim), device=device)
        
        self.a = torch.zeros((size,t_episode, a_dim), device=device)
        
        self.r_in = torch.zeros((size,t_episode, r_dim), device=device)
        
        self.mask = torch.zeros((size,t_episode, mask_dim), device=device)
        self.future_p = 1 - (1. / (1 + replay_k))
        self.cnt = 0

        self.meta = meta
        
    def store(self, s, a, r_in, mask, s_):
        s, a, mask, s_ = torch.tensor(s),torch.tensor(a),torch.tensor(mask),torch.tensor(s_)
        s, a, mask, s_ = s.to(self.device).reshape(-1), a.to(self.device).reshape(-1), mask.to(self.device), s_.to(self.device).reshape(-1)
        s, s_ = s.detach(), s_.detach()
        index = (self.cnt // t_episode) % self.size
        t_step = self.cnt % t_episode
        self.sobs[index,t_step] = s[:self.obs_dim]
        self.sag[index,t_step] = s[self.obs_dim:-self.dg_dim]
        self.sdg[index,t_step] = s[-self.dg_dim:]
        
        self.a[index,t_step] = torch.tensor(a)
        self.r_in[index,t_step] = torch.tensor(r_in)
        
        self.mask[index,t_step] = mask
        
        self.s_obs[index,t_step] = s_[:self.obs_dim]
        self.s_ag[index,t_step] = s_[self.obs_dim:-self.dg_dim]
        self.s_dg[index,t_step] = s_[-self.dg_dim:]
        self.cnt += 1
        return
        
    def sample(self, batch_size):

        # Select which episodes and time steps to use.
        right = min(self.cnt//self.t_episode - 1, self.size)
        episode_idxs = np.random.randint(0, right, batch_size)
        t_samples = np.random.randint(self.t_episode, size=batch_size)
        
        sobs = self.sobs[episode_idxs, t_samples].clone()
        sag = self.sag[episode_idxs, t_samples].clone()
        sdg = self.sdg[episode_idxs, t_samples].clone()
        
        a = self.a[episode_idxs, t_samples].clone()
        r_in = self.r_in[episode_idxs, t_samples].clone()
        r_env = torch.zeros_like(r_in)
        
        mask = self.mask[episode_idxs, t_samples].clone()
        
        s_obs = self.s_obs[episode_idxs, t_samples].clone()
        s_ag = self.s_ag[episode_idxs, t_samples].clone()
        s_dg = self.s_dg[episode_idxs, t_samples].clone()

        #print("==========before=========")
        #print("sobs sag sdg")
        #print(sobs,sag,sdg)
        #print("a mask r")
        #print(a,mask)
        #print("s_obs s_ag s_dg")
        #print(s_obs,s_ag,s_dg)

        if not self.meta:
            # Hindsight Experience Replay
            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = np.random.uniform(size=batch_size) * (self.t_episode - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.
            #print("t_samples")
            #print(t_samples)
            #print("future_offset")
            #print(future_offset)
            #print("her_indexes, future_t")
            #print(her_indexes, future_t)
            future_ag = self.sag[episode_idxs[her_indexes], future_t]
            #print("future_ag")
            #print(future_ag)
            sdg[her_indexes] = future_ag
            s_dg[her_indexes] = future_ag
            r_env = self.reward(s_ag, s_dg)
        else:
            r_in += -1 
        
        s = torch.cat([sobs, sag, sdg], 1)
        s_ = torch.cat([s_obs, s_ag, s_dg], 1)
        #print("==========after=========")
        #print("sobs sag sdg")
        #print(sobs,sag,sdg)
        #print("a mask r")
        #print(a,mask,r)
        #print("s_obs s_ag s_dg")
        #print(s_obs,s_ag,s_dg)
        
        r = r_env + r_in
        assert r.shape[1] == 1
        s, a, r, mask, s_ = s.to(self.odevice), a.to(self.odevice), r.to(self.odevice), mask.to(self.odevice), s_.to(self.odevice)
        return s, a, r, mask, s_
        
class HashTable(object):
    def __init__(self, k, dim):
        self.A = np.random.randn(k, dim)
        self.dic = dict()

    def add(self, s, a):
        code = self.getcode(s, a)
        if code not in self.dic.keys():
            self.dic[code] = 1
        else:
            self.dic[code] += 1

    def freq(self, s, a):
        code = self.getcode(s, a)
        if code not in self.dic.keys():
            return 0
        else:
            return self.dic[code]

    def getcode(self, s, a):
        if type(s) == torch.Tensor and (not s.device==torch.device(type='cpu')):
            s = s.cpu()
        if type(a) == torch.Tensor and (not a.device==torch.device(type='cpu')):
            a = a.cpu()
        arr = np.concatenate([np.array(s).reshape(-1),np.array(a).reshape(-1)]).reshape(-1,1)
        code = tuple(np.sign(self.A @ arr).flatten())
        return code
        
class Normalizer(object):
    def __init__(self, dim, cliprange, device):
        self.sum = torch.zeros(dim,requires_grad=False, device=device)
        self.sumsq = torch.zeros(dim,requires_grad=False, device=device)
        self.dim = dim
        self.cnt = 0
        self.eps = torch.tensor(1e-3, device=device)
        self.cliprange = cliprange

    def update(self, x):
        self.sum += x.sum(dim=0)
        self.sumsq += (x**2).sum(dim=0)
        self.cnt += x.shape[0]

    def view(self, mean):
        return mean.view((1, self.dim))

    def get_mean_std(self):
        mean = self.view(self.sum/self.cnt)
        var = self.sumsq/self.cnt - (self.sum/self.cnt)**2
        var = torch.max(self.eps, var)
        std = self.view((var)**0.5)
        return mean, std

    def normalize(self, x):
        mean, std = self.get_mean_std()
        x = ((x-mean)/std).clamp(-self.cliprange, self.cliprange)
        return x

    def denormalize(self, x):
        mean, std = self.get_mean_std()
        x = (x*std + mean).clamp(-self.cliprange, self.cliprange)
        return x

class μNet(nn.Module):
    def __init__(self, norm, N_STATES, N_ACTIONS):
        super(μNet, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(N_STATES, 300)
        self.fc1.weight.data.normal_(0, 1e-3)   # initialization
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 1e-3)   # initialization
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 1e-3)   # initialization
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 1e-3)   # initialization
        self.fc5 = nn.Linear(300, 10)
        self.fc5.weight.data.normal_(0, 1e-3)   # initialization
        self.out = nn.Linear(300, N_ACTIONS)
        self.out.weight.data.normal_(0, 1e-3)   # initialization
        self.fcn = nn.Linear(N_ACTIONS, N_ACTIONS)
        self.fcn.weight.data.normal_(0, 1e-3)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2)
        

    def forward(self, state):
        sl = self.norm.normalize(state)
        x = self.act(self.fc1(sl))
        x = self.act(self.fc2(x))
        #x = self.act(self.fc3(x))
        #x = self.act(self.fc4(x))
        #x = self.act(self.fc5(x))
        x = self.out(x)
        noise = self.fcn(torch.rand_like(x)).clamp(-NOISE_CLIP, NOISE_CLIP)
        x += noise
        al = self.tanh(x)
        return al

class QNet(nn.Module):
    def __init__(self, norm, N_STATES, N_ACTIONS):
        super(QNet, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 300)
        self.fc1.weight.data.normal_(0, 1e-3)   # initialization
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 1e-3)   # initialization
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 1e-3)   # initialization
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 1e-3)   # initialization
        self.fc5 = nn.Linear(300, 300)
        self.fc5.weight.data.normal_(0, 1e-3)   # initialization
        self.out = nn.Linear(300, 1)
        self.out.weight.data.normal_(0, 1e-3)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, state, al):
        sl = self.norm.normalize(state)
        x = torch.cat([sl,al],dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        #x = self.act(self.fc3(x))
        #x = self.act(self.fc4(x))
        #x = self.act(self.fc5(x))
        #print("tanh:",self.tanh(self.out(x))[0].item())
        Q = self.tanh(self.out(x))/(1-γ)
        #print("out:",out[0].item())
        return Q

class PNet(nn.Module):
    def __init__(self, norm, N_STATES, N_ACTIONS):
        super(PNet, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(N_STATES+N_ACTIONS, 300)
        self.fc1.weight.data.normal_(0, 1e-3)   # initialization
        self.fc2 = nn.Linear(300, 300)
        self.fc2.weight.data.normal_(0, 1e-3)   # initialization
        self.fc3 = nn.Linear(300, 300)
        self.fc3.weight.data.normal_(0, 1e-3)   # initialization
        self.fc4 = nn.Linear(300, 300)
        self.fc4.weight.data.normal_(0, 1e-3)   # initialization
        self.fc5 = nn.Linear(300, 300)
        self.fc5.weight.data.normal_(0, 1e-3)   # initialization
        self.out = nn.Linear(300, N_STATES)
        self.out.weight.data.normal_(0, 1e-3)   # initialization
        self.tanh = nn.Tanh()
        self.act = nn.LeakyReLU(0.2)
        

    def forward(self, x1, x2):
        #x1 = self.norm.normalize(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        #x = self.act(self.fc4(x))
        #x = self.act(self.fc5(x))
        x = self.out(x)
        #out = self.norm.denormalize(x)
        return out

class TD3(Policy):
    def __init__(self, env, states, actions, robot, reward, device):
        super(TD3, self).__init__(states, actions)
        STATES_SHAPE = [i.shape[0] for i in states()]
        print(STATES_SHAPE)
        N_STATES = sum(STATES_SHAPE)
        N_ACTIONS = actions.space.sample().shape[0]
        print(N_ACTIONS)

        self.robot = robot
        self.reward = reward
        self.env = env
        self.states = states
        self.actions = actions
        self.action_data = None
        self.device = device
        self.high = torch.tensor(actions.space.high, device=self.device)
        self.low = torch.tensor(actions.space.low, device=self.device)
        print(self.high, self.low)
        self.norm = Normalizer(N_STATES,1e3, device)
        self.μ = μNet(self.norm, N_STATES, N_ACTIONS)
        self.μ.to(device)
        self.Q1, self.Q2 = QNet(self.norm, N_STATES, N_ACTIONS), QNet(self.norm, N_STATES, N_ACTIONS)
        self.Q1.to(self.device)
        self.Q2.to(self.device)
        #self.P = PNet(self.norm)
        #self.P.to(device)
        trainableμ = list(filter(lambda p: p.requires_grad, self.μ.parameters()))
        self.optimμ = torch.optim.Adam(trainableμ, lr=LR)
        trainableQ1 = list(filter(lambda p: p.requires_grad, self.Q1.parameters()))
        self.optimQ1 = torch.optim.Adam(trainableQ1, lr=LR)
        trainableQ2 = list(filter(lambda p: p.requires_grad, self.Q2.parameters()))
        self.optimQ2 = torch.optim.Adam(trainableQ2, lr=LR)
        #trainableP = list(filter(lambda p: p.requires_grad, self.P.parameters()))
        #self.optimP = torch.optim.Adam(trainableP, lr=LR)
        self.μ_tar = μNet(self.norm, N_STATES, N_ACTIONS)
        self.Q1_tar, self.Q2_tar = QNet(self.norm, N_STATES, N_ACTIONS), QNet(self.norm, N_STATES, N_ACTIONS)
        self.μ_tar.to(device)
        self.Q1_tar.to(self.device)
        self.Q2_tar.to(self.device)
        self.μ_tar.load_state_dict(self.μ.state_dict())
        self.Q1_tar.load_state_dict(self.Q1.state_dict())
        self.Q2_tar.load_state_dict(self.Q2.state_dict())

        self.lsh_reward = lambda s,a,s_: intrinsic_counting_reward(s_,a) * β #+ torch.mean((self.P(s, a)-s_)**2)* p_ratio
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE,N_STATES-6, 3, 3, N_ACTIONS, 1, 1, t_episode, REPLAY_K, reward, False, device, device)
        self.meta_replay_buffer = ReplayBuffer(BUFFER_SIZE, N_STATES-6, 3, 3, N_ACTIONS, 1, 1, t_episode, REPLAY_K, reward, True, device, device)
        self.cnt_step = 0
        self.lossμ, self.lossQ1, self.lossQ2 = 0,0,0

    def learn(self, done):
        self.lossμ, self.lossQ1, self.lossQ2 = 0,0,0
        
        # get state $s_t$
        st = self.states.vec_torch_data.float().to(self.device).unsqueeze(0)
        self.norm.update(st)
        
        # get action $a_t$
        at_np = self.get_action(st)
        at = torch.tensor(at_np, device=self.device).unsqueeze(0)

        # apply action
        self.set_action_data(at_np)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        tik = time.time()
        st_, _, _, info = self.env.step()
        tok = time.time()
        st_ = torch.tensor(st_, dtype=torch.float,device=self.device).unsqueeze(0)

        # add to hash table and compute intrinsic reward
        hashtable.add(st_, at)

        # take last but one state as achieved goal, take last state as desired goal
        sag = torch.tensor(self.states()[-2]).unsqueeze(0)
        sdg = torch.tensor(self.states()[-1]).unsqueeze(0)
        rt_env = self.reward(sag, sdg)[0].item()
        rt_lsh = self.lsh_reward(st, at, st_)
        rt_sim = (tok - tik) * sim_ratio
        rt_in = rt_lsh + rt_sim

        #with torch.no_grad():
        #    ai_ = self.μ_tar(st_)
        #    for i in range(len(self.high)):
        #        ai_[:, i] = ai_[:, i].clamp(-self.low[i], self.high[i])
        #
        #    Q1_ = self.Q1_tar(st_,ai_)
        #    Q2_ = self.Q2_tar(st_,ai_)
        #q = self.Q1(st, at)
        #if self.cnt_step % t_episode == 0:
        #    print("epi{:d}".format(self.cnt_step//t_episode),flush=True)
        #print("Q:{:.3f}\tQ*:{:.3f}\trt:{:.3f}".format(q.item(),Q1_.item(),rt_env))
        
        self.replay_buffer.store(st, at, rt_in, 0. if done else γ, st_)
        # keep (st, at, rt_in, st_) in meta training buffer
        if self.cnt_step < start_timesteps:
            self.meta_replay_buffer.store(st, at, rt_in, 0. if done else γ, st_)
        
        if self.cnt_step > 2*t_episode and (self.cnt_step % DELAY_CRITIC_STEPS == 0 or self.cnt_step % DELAY_ACTOR_STEPS == 0):
            for _ in range(N_BATCHES):
                self.train()
            
        if self.cnt_step % TARGET_REPLACE_STEPS == 0:
            self.update_target()
            
        self.cnt_step += 1
        return rt_env,rt_lsh,rt_sim, self.lossμ*(DELAY_ACTOR_STEPS//t_episode)/N_BATCHES, self.lossQ1*(DELAY_CRITIC_STEPS//t_episode)/N_BATCHES, self.lossQ2*(DELAY_CRITIC_STEPS//t_episode)/N_BATCHES
        
    def update_target(self):
        update_pairs = [(self.μ, self.μ_tar), (self.Q1, self.Q1_tar), (self.Q2, self.Q2_tar)]
        for i, i_tar in update_pairs:
            for param, target_param in zip(i.parameters(), i_tar.parameters()):
                target_param.data.copy_(polyak * param.data + (1 - polyak) * target_param.data)
        
    def train(self):
        if self.cnt_step >= start_timesteps:
            # randomly sample from memory
            s,a,r,mask,s_ = self.replay_buffer.sample(BATCH_SIZE)
        else:
            s,a,r,mask,s_ = self.meta_replay_buffer.sample(BATCH_SIZE)
        if self.cnt_step % DELAY_CRITIC_STEPS == 0:
            lq1, lq2 = self.update_critic(s,a,r,mask,s_)
            self.lossQ1 += lq1
            self.lossQ2 += lq2
            #self.rt_p = self.update_physical_predictor(s, a, s_).item()
        
        if self.cnt_step % DELAY_ACTOR_STEPS == 0:
            self.lossμ += self.update_actor(s)
        return
            
    def action_logits(self, a):
        # rescale action to [-1,1]
        return 2*(a - self.low)/(self.high - self.low) - 1

    def logits_action(self, al):
        return (al+1) * (self.high - self.low)/2 + self.low
        
    def get_action(self, st):
        # cast to numpy
        if np.random.uniform() < RAND_EPSILON:
            at_np = self.env.action.space.sample()
        else:
            # compute action with actor μ
            with torch.no_grad():
                alt = self.μ(st)
            #noise = (torch.randn_like(alt) * NOISE_SCALE).clamp(-NOISE_CLIP, NOISE_CLIP)
            #alt += noise
            #alt = alt.clamp(-1,1)
            at = self.logits_action(alt)
            #for i in range(len(self.high)):
            #    at[:, i] = at[:, i].clamp(-self.low[i], self.high[i])
            at_np = at.detach().cpu().numpy()[0]
            #print("{:d}".format(self.cnt_step % t_episode))
            #print(at_np)
        return at_np
        
    def update_physical_predictor(self, s, a, s_):
        s_p = self.P(s, a)
        lossP = torch.mean((s_p-s_)**2)
        lossP.backward()
        self.optimP.step()
        self.optimP.zero_grad()
        return lossP * p_ratio

    def update_actor(self, si):
        ali = self.μ(si)
        Q1 = self.Q1(si, ali)
        lossμ = -torch.mean(Q1) + ACTION_L2 * torch.mean(ali**2)
        lossμ.backward()
        #print("out layer grad:",self.μ.out.weight.grad)
        #print("fcn layer:",self.μ.fcn.weight)
        self.optimμ.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        return lossμ.item()
        
    def update_critic(self,si,ai,ri,mask,si_):
        #print("si:",si[0])
        #print("ai:",ai[0])
        #print("si_:",si_[0])
        #print("ri:",ri)
        #print("mask:",mask[0])
        with torch.no_grad():
            ali_ = self.μ_tar(si_)
        
            Q1_ = self.Q1_tar(si_,ali_)
            Q2_ = self.Q2_tar(si_,ali_)
            y = ri + mask * torch.min(Q1_, Q2_)
        #    print("Q_:",torch.min(Q1_, Q2_)[0])

        #print("Q*:",y)
        ali = self.action_logits(ai)

        Q1 = self.Q1(si, ali) 
        #print("Q1:",Q1,flush=True)
        lossQ1 = torch.mean((y-Q1)**2)
        lossQ1.backward()
        self.optimQ1.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()

        Q2 = self.Q2(si, ali)
        lossQ2 = torch.mean((y-Q2)**2)
        lossQ2.backward()
        self.optimQ2.step()
        self.optimμ.zero_grad()
        self.optimQ1.zero_grad()
        self.optimQ2.zero_grad()
        return lossQ1.item(),lossQ2.item()

    def save(self, filename, normfile):
        torch.save({'μ':self.μ_tar.state_dict(),
                    'Q1':self.Q1_tar.state_dict(),
                    'Q2':self.Q2_tar.state_dict()}, filename)
        with open(normfile, "wb") as f:
            pickle.dump(self.norm, f)
    
    def load(self, filename, normfile):
        state_dicts = torch.load(filename)
        self.μ.load_state_dict(state_dicts['μ'])
        self.Q1.load_state_dict(state_dicts['Q1'])
        self.Q2.load_state_dict(state_dicts['Q2'])
        self.μ_tar.load_state_dict(state_dicts['μ'])
        self.Q1_tar.load_state_dict(state_dicts['Q1'])
        self.Q2_tar.load_state_dict(state_dicts['Q2'])
        with open(normfile, "rb") as f:
            self.norm = pickle.load(f)
            self.μ.norm = self.norm
def success(reward):
    return reward >= -0.5

if __name__=="__main__":
    sim = prl.simulators.Bullet(render=False)
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0.2),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    end_effector = manipulator.get_end_effector_ids()[-1]
    other_links = manipulator.get_end_effector_ids()[:-1] + manipulator.get_link_ids()
    states = LinkWorldVelocityState(manipulator, link_ids=end_effector) + LinkWorldPositionState(manipulator, link_ids=other_links)  + LinkWorldVelocityState(manipulator, link_ids=other_links) + LinkWorldPositionState(manipulator, link_ids=end_effector) + PositionState(box,world)
    STATES_SHAPE = [i.shape[0] for i in states()]
    N_STATES = sum(STATES_SHAPE)
    action = JointPositionChangeAction(manipulator)
    N_ACTIONS = action.space.sample().shape[0]
    env = Env(world, states,actions=action)
    
    env.reset()
    
    hashtable = HashTable(LSH_K, N_STATES+N_ACTIONS)
    td3 = TD3(env, states, action,manipulator,touched_reward, "cuda")
    
#    print("Loading model...")
#    td3.load("td3.ckpt", "norm.pickle")
    save_freq = 200
    
    n_cycles = 10
    n_success = 0
    s_reward = 0
    s_lsh = 0
    s_sim = 0
    sum_lossμ = 0
    sum_lossQ1 = 0
    sum_lossQ2 = 0
    tik = time.time()
    for i in count(start=1):
        # randomly reset robot joint and box position
        manipulator.reset_joint_states(np.random.uniform(-1,1,(15)))
        world.move_object(box,[np.random.uniform(-1,1),np.random.uniform(-0.5,0.5),0.2])
        
        # run an episode
        for t in range(t_episode):
            done = (t >= t_episode - 1)
            reward,r_lsh,r_sim, lossμ, lossQ1, lossQ2 = td3.learn(done)
            sum_lossμ += lossμ
            sum_lossQ1 += lossQ1
            sum_lossQ2 += lossQ2
            s_reward += reward
            s_lsh += r_lsh
            s_sim += r_sim
        n_success += 1 if success(reward) else 0
        #print("SUCCESS" if done else "FAIL",flush=True)
        
        # collect statistics of #n_cycles results
        if i%n_cycles==0:
            tok = time.time()
            print("Epoch {:5d}\tSuc rate: {:.2f}\tAvg reward: {:.2f}\tLSH reward: {:.5f}\tSim reward: {:.5f}\tLossμ:{:.3f}\tLossQ1:{:.3f}\tLossQ2:{:.3f}\tTime: {:.1f}".format(int(i/n_cycles),n_success/n_cycles,s_reward/n_cycles,s_lsh/n_cycles, s_sim/n_cycles, sum_lossμ/n_cycles, sum_lossQ1/n_cycles, sum_lossQ2/n_cycles, tok-tik),flush=True)
            sum_lossμ = 0
            sum_lossQ1 = 0
            sum_lossQ2 = 0
            n_success = 0
            s_reward = 0
            s_lsh = 0
            s_sim = 0
            tik = time.time()
        if i%save_freq==0:
#            print("Saving model...")
            td3.save("td3.ckpt", "norm.pickle")
#    for i in count():
#        obs,reward,done,info = env.step()
#        print(reward)
