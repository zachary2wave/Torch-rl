
import torch
import random
import numpy as np
from abc import ABC
from copy import deepcopy
from collections import deque

class Memory(ABC):

    def __init__(self, capacity, other_record=None):
        self.capacity = capacity
        self.memory = {"s":  deque(maxlen=capacity),
                       "a":  deque(maxlen=capacity),
                       "s_": deque(maxlen=capacity),
                       "r":  deque(maxlen=capacity),
                       "tr": deque(maxlen=capacity)}
        if other_record is not None:
            for key in other_record:
                self.memory[key] = []
        self.position = 0

    def push(self, sample):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

class ReplayMemory(Memory):
    def __init__(self, capacity, other_record=None):
        super(ReplayMemory, self).__init__(capacity, other_record=other_record)

    def push(self, sample):
        """Saves a transition."""
        for key in self.memory.keys():
            self.memory[key].append(sample[key])
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

    def recent_step_sample(self, batch_size):
        sample = {"s": [], "a": [], "s_": [], "r": [], "tr": []}
        for key in self.memory.keys():
            sample[key] = self.memory[key][-batch_size:]
            sample[key] = np.array(sample[key], dtype=np.float32)
            sample[key] = torch.from_numpy(sample[key])
        return sample

    def __len__(self):
        return len(self.memory)


class ReplayMemory_HIRO(Memory):
    def __init__(self, capacity, other_record=None):
        super(ReplayMemory, self).__init__(capacity, other_record)
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


class ReplayMemory_Sequence():
    def __init__(self, capacity, max_seq_len, other_record=None):
        self.capacity = capacity
        Sequence = {"s": [],
                    "a": [],
                    "s_": [],
                    "r": [],
                    "tr": []}
        if other_record is not None:
            for key in other_record:
                Sequence[key] = []
        self.Sequence = Sequence
        self.memory = [deepcopy(Sequence)]
        self.position = 0
        self.max_position = 0
        self.max_seq_len = max_seq_len
        self.batch = 32
        self.sequence_len = 100

    def push(self, sample):
        """Saves a transition."""
        for key in self.memory[self.position].keys():
            if sample[key] is np.ndarray:
                self.memory[self.position][key].append(sample[key])
            else:
                self.memory[self.position][key].append(np.array(sample[key]))
        if sample["tr"] == 1:
            self.position = (self.position + 1) % self.capacity
            if self.max_position <= self.capacity:
                self.memory.append(deepcopy(self.Sequence))
                self.max_position += 1

    def sample_ep(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        sample_index = random.sample(range(self.max_position), self.batch_size)
        sample = {}
        for key in self.Sequence.keys():
            sample[key] = torch.empty((self.max_seq_len, self.batch_size, self.memory[0][key][0].shape[0]), dtype=torch.float32)
            for flag, index in enumerate(sample_index):
                ep_len = len(self.memory[index]['s'])
                for time_step in range(self.max_seq_len):
                    if ep_len > self.max_seq_len:
                        print("the memory size is longer than max_seq_len")
                        sample[key][time_step, flag, :] = torch.from_numpy(self.memory[index][key][time_step])
                    elif ep_len == self.max_seq_len:
                        sample[key][time_step, flag, :] = torch.from_numpy(self.memory[index][key][time_step])
                    else:
                        for time_step in range(ep_len):
                            sample[key][time_step, flag, :] = torch.from_numpy(self.memory[index][key][time_step])
                        for time_step in range(ep_len, self.max_seq_len):
                            sample[key][time_step, flag, :] = torch.zeros(size=len(self.memory[index][key][0]))
        return sample

    def sample_sequence(self, batch_size=None, sequence_len=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if sequence_len is not None:
            self.sequence_len = sequence_len
        sample = {}
        for key in self.Sequence.keys():
            temp_len = self.memory[0][key][0].size
            sample[key] = torch.empty((self.sequence_len, self.batch_size, temp_len), dtype=torch.float32)
            for loop in range(self.batch_size):
                index = random.sample(range(self.max_position), 1)[0]
                ep_len = len(self.memory[index]['s'])
                if ep_len <= self.sequence_len:
                    for time_step in range(ep_len):
                        sample[key][time_step, loop, :] = \
                            torch.from_numpy(self.memory[index][key][time_step])
                    for time_step in range(ep_len, self.sequence_len):
                        sample[key][time_step, loop, :] = torch.zeros(temp_len)
                else:
                    start_ = random.sample(range(0, ep_len - self.sequence_len), 1)[0]
                    end_ = start_ + self.sequence_len
                    for (time_step, time) in enumerate(range(start_, end_)):
                        sample[key][time_step, loop, :] = \
                            torch.from_numpy(self.memory[index][key][time_step])
        return sample

    def recent_ep_sample(self):
        return self.memory[self.position]

    def __len__(self):
        return len(self.memory)