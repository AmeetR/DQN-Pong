import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
import random


Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """
    This class implements DQN with conv layers. The idea is to take an impage of the Pong
    game state as input and output a probability vector over actions

    The network has 3 conv layers, each with a kernel size of 8, 4, and 3. The input is
    a 4 channel image of size 84x84. The output is a vector of 14 probabilities over
    the 14 possible actions.

    Args:
        in_channels (int): number of channels in the input image
        n_actions (int): number of possible actions
    """
    def __init__(self, in_channels = 4, n_actions = 4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)



    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



