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

from myenv import *
t_episode = 100
        
        

class TD3_test(TD3):
    def __init__(self, env, states, actions, robot, reward, device):
        super(TD3_test, self).__init__(self, env, states, actions, robot, reward, device)

    def learn(self):
        # get state $s_t$
        st = self.states.vec_torch_data.float().to(self.device).unsqueeze(0)
        
        # get action $a_t$
        at_np = self.get_action(st)
        at = torch.tensor(at_np, device=self.device).unsqueeze(0)

        # apply action
        self.set_action_data(at_np)
        self.actions()
        
        # step in environment to get next state $s_{t+1}$, reward $r_t$
        self.env.step(1/240.0)
        # take last but one state as achieved goal, take last state as desired goal

        return 
        
    def get_action(self, st):
        # cast to numpy
        # compute action with actor μ
        with torch.no_grad():
            alt = self.μ(st)
        noise = (torch.randn_like(alt) * NOISE_SCALE).clamp(-NOISE_CLIP, NOISE_CLIP)
        alt += noise
        alt = alt.clamp(-1,1)
        at = self.logits_action(alt)
        #for i in range(len(self.high)):
        #    at[:, i] = at[:, i].clamp(-self.low[i], self.high[i])
        at_np = at.detach().cpu().numpy()[0]
        #print("{:d}".format(self.cnt_step % t_episode))
        #print(at_np)
        return at_np

if __name__=="__main__":
    sim = prl.simulators.Bullet(render=True)
    world = prl.worlds.BasicWorld(sim)
    box = world.load_box(position=(0.5,0,0.2),dimensions=(0.1,0.1,0.1),mass=0.1,color=[0,0,1,1])
    manipulator = world.load_robot('wam')
    end_effector = manipulator.get_end_effector_ids()[-1]
    other_links = manipulator.get_end_effector_ids()[:-1] + manipulator.get_link_ids()
    states = LinkWorldVelocityState(manipulator, link_ids=end_effector) + LinkWorldPositionState(manipulator, link_ids=other_links)  + LinkWorldVelocityState(manipulator, link_ids=other_links) + LinkWorldPositionState(manipulator, link_ids=end_effector) + PositionState(box,world)
    STATES_SHAPE = [i.shape[0] for i in states()]
    print(STATES_SHAPE)
    N_STATES = sum(STATES_SHAPE)
    action = JointPositionAction(manipulator)
    N_ACTIONS = action.space.sample().shape[0]
    print(N_ACTIONS)
    env = Env(world, states,actions=action)
    
    env.reset()
    
    hashtable = HashTable(LSH_K, N_STATES+N_ACTIONS)
    td3 = TD3(env, states, action,manipulator,touched_reward, "cuda")
    
    print("Loading model...")
    td3.load("td3.ckpt", "norm.pickle")
    
    for i in count(start=1):
        # randomly reset robot joint and box position
        manipulator.reset_joint_states(np.random.uniform(-1,1,(15)))
        world.move_object(box,[np.random.uniform(-1,1),np.random.uniform(-0.5,0.5),0.2])
        
        # run an episode
        for t in range(t_episode):
            td3.learn()
