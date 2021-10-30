import copy
from collections import namedtuple
from itertools import count

import math
import random
import numpy as np
import time

import gym

from networks.Q_networks import DQN, ReplayMemory
from wrapppers.atari_wrappers import *
from utils import get_state, select_action

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))






 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    


    state_batch = torch.cat(batch.state).cuda()
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(env, n_episodes, render=False):
    
    env = gym.wrappers.Monitor(env, video_path + 'dqn_pong_video', force = True, video_callable=lambda episode_id: episode_id%100==0)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 5 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
                torch.save(policy_net, video_path + "dqn_pong_model.pth")
    env.close()
    return policy_net


if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    video_path = './train_videos2/'
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    TRAIN_RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 100000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    policy_net = DQN().to(device)
    print(policy_net)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    policy_net = train(env, 10000, render = TRAIN_RENDER)
    torch.save(policy_net, "dqn_pong_model.pth")
    print("Model saved")

    
    



    