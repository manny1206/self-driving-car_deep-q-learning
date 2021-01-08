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

###### Architecture of the neural network ######
class Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # full connection from input to hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        # full connection from hidden layer to output
        self.fc2 = nn.Linear(30, output_size)
        
    def prop_forward(self, state):
        x = F.relu(self.fc1(state))
        q_vals = self.fc2(x)
        
        return q_vals

###### Experience Replay... ######
# consider the past n = 100 remembered states to decide an action
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # memory for past states

    # function to push an event to the replay memory
    def push(self, event):
        self.memory.append(event)
        # make sure the memory doesn't exceed the capacity
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    # function to randomly sample from the memory
    def random_sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

###### Implement Deep Q-Learning ######

class Dqn():
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        self.reward_window =  []
        self.model = Network(input_size, output_size)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.prev_state = torch.Tensor(input_size).unsqueeze(0)
        self.prev_output = 0
        self.prev_reward = 0