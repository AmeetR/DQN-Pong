from itertools import count

import numpy as np
import time
import torch

import gym

from networks.Q_networks import DQN, ReplayMemory
from wrapppers.atari_wrappers import *

from utils import get_state, select_action


def test(env, n_episodes, render=True):

    env = gym.wrappers.Monitor(env, SAVE_DIR + 'dqn_pong_video', force = True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            with torch.no_grad():
                action = policy_net(state.cuda()).max(1)[1].view(1, 1)
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return


if __name__ == '__main__':


    env = gym.make('PongNoFrameskip-v4')
    env = make_env(env)
    env.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = 'train_videos2/dqn_pong_model.pth'
    SAVE_DIR = './videos/'

    
    policy_net = torch.load(weights_path).to(device)
    print(policy_net)

    TEST_RENDER = True


    test(env, 2, render = TEST_RENDER)

    