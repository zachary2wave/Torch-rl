
import torch
import random
import numpy as np
from collections import namedtuple


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {"s":[],"a":[],"s_":[],"r":[],"tr":[]}
        keylist = self.memory.keys()
        self.position = 0

    def push(self, sample):
        """Saves a transition."""
        for key in self.memory.keys():
            self.memory[key].append(sample[key])
        if len(self.memory["s"]) > self.capacity:
            for key in self.memory.keys():
                del self.memory[key][0]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample_index = random.sample(range(len(self.memory["s"])), batch_size)
        sample = {"s": [], "a": [], "s_": [], "r": [], "tr": []}
        for key in self.memory.keys():
            for index in sample_index:
                sample[key].append(self.memory[key][index])
            sample[key] = np.array(sample[key],dtype=np.float32)
            sample[key] = torch.from_numpy(sample[key])
        return sample

    def extract_last_ep(self, during):
        sample_index = np.arange(-during,-1)
        sample = dict()
        for key in self.memory.keys():
            sample[key] = self.memory[key][sample_index]
        return sample

    def __len__(self):
        return len(self.memory)