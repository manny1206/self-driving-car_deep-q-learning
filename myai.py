# import libraries
import numpy as np
import random
import os

# torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# architecture of the neural network
class Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, 30)
        
        self.fc2 = nn.Linear(30, output_size)
        
    def prop_forward(self, state):
        x = F.relu(self.fc1(state))
        q_vals = self.fc2(x)
        
        return q_vals

# Experience Replay...
# consider the past n = 100 remembered states to decide an action
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # memory for past states