
import torch
import random
import numpy as np
from abc import ABC


class Memory(ABC):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {"s":[],"a":[],"s_":[],"r":[],"tr":[]}
        keylist = self.memory.keys()
        self.position = 0

    def push(self, sample):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

class ReplayMemory(Memory):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(capacity)

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


class Sequence_Replay_Memory(Memory):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(capacity)
        for key in self.memory.keys():
            self.memory[key].append([])

    def push(self, sample):
        for key in self.memory.keys():
            self.memory[key][-1].append(sample[key])
        if sample["tr"]:
            for key in self.memory.keys():
                self.memory[key].append([])
                del self.memory[key][0]

    def sample(self, batch_size):
        sample_index = random.sample(range(len(self.memory["s"])), batch_size)
        sample = {"s": [], "a": [], "s_": [], "r": [], "tr": []}
        for key in self.memory.keys():
            for index in sample_index:
                temp = np.array(self.memory[key][index], dtype=np.float32)
                sample[key].append(torch.from_numpy(temp))
        return sample

class ReplayMemory_HIRO(Memory):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(capacity)
        self.memory = {"s": [],"g":[], "a": [], "s_": [], "r": [], "tr": []}
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
        for key in sample.keys():
            for index in sample_index:
                if key == "s":
                    temp = np.array(self.memory["s"][index]+self.memory["g"][index], dtype=np.float32)
                    sample[key].append(torch.from_numpy(temp))
                else:
                    temp = np.array(self.memory[key][index], dtype=np.float32)
                    sample[key].append(torch.from_numpy(temp))
        return sample
    def H_sample(self, batch_size):
        sample_index = random.sample(range(len(self.memory["s"])), batch_size)
        sample = {"s": [], "g": [], "s_": [], "r": [], "tr": []}
        for key in sample.keys():
            for index in sample_index:
                temp = np.array(self.memory[key][index], dtype=np.float32)
                sample[key].append(torch.from_numpy(temp))
        return sample
