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
        self.model = Network(input_size, output_size)
        
        self.gamma = gamma
        self.reward_window =  []
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.prev_state = torch.Tensor(input_size).unsqueeze(0)
        self.prev_output = 0
        self.prev_reward = 0

    # use softmax to choose max q-value,
    # while generating probability distributions for each output q-value,
    # to choose the output action
    def choose_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7) # 7 = temperature to exaggerate probabilities
        action = probs.multinomial()
        return action.data[0,0]

    # a function that adjusts the weights of the neural network transitions,
    # using a sample of states (aka. batch)
    def q_learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        # calculate loss
        td_loss = F.smooth_l1_loss(outputs, target)
        # ?
        self.optimizer.zero_grad()
        # back propogate through the network
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    def update(self, last_reward, last_signal):
        #remember that a state is the signal as a tensor
        new_state = torch.Tensor(last_signal).float().unsqueeze(0)
        self.memory.push(self.prev_state, new_state, torch.LongTensor([int(self.prev_output)]), 
        torch.Tensor([self.prev_reward)])