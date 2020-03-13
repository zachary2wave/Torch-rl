
import torch
import random
import numpy as np
from abc import ABC


class Memory(ABC):

    def __init__(self, capacity, other_record=None):
        self.capacity = capacity
        self.memory = {"s": [], "a": [], "s_" : [], "r": [], "tr": []}
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
        Sequence = {"s": np.zeros(),
                    "a": np.zeros(max_seq_len),
                    "s_": np.zeros(max_seq_len),
                    "r": np.zeros(max_seq_len),
                    "tr": np.zeros(max_seq_len)}
        if other_record is not None:
            for key in other_record:
                Sequence[key] = np.zeros(max_seq_len)
        self.memory = [Sequence]*self.capacity
        self.position = 0
        self.max_position = 0
        self.max_seq_len = max_seq_len
        self.batch = batch_size
        self.sequence_len = sequence_len

    def push(self, sample):
        """Saves a transition."""
        for key in self.memory.keys():
            self.memory[self.position][key].append(sample[key])
        if sample["tr"] == 1:
            self.position = (self.position + 1) % self.capacity
            if self.max_position <= self.capacity:
                self.max_position += 1

    def sample_ep(self, batch_size = None):
        if batch_size is not None:
            self.batch_size = batch_size
        sample_index = random.sample(range(self.max_position), batch_size)
        sample = {"s":  torch.empty((self.max_seq_len, batch_size, len(self.memory[0]['s'])), dtype=torch.float32),
                  "a":  torch.empty((self.max_seq_len, batch_size, len(self.memory[0]['a'])), dtype=torch.float32),
                  "s_": torch.empty((self.max_seq_len, batch_size, len(self.memory[0]['s'])), dtype=torch.float32),
                  "r":  torch.empty((self.max_seq_len, batch_size, 1), dtype=torch.float32),
                  "tr": torch.empty((self.max_seq_len, batch_size, 1), dtype=torch.float32)}
        for flag, index in enumerate(sample_index):
            ep_len = len(self.memory[index]['s'])
            for key in self.memory.keys():
                for time_step in range(self.max_seq_len):
                    if ep_len > self.max_seq_len:
                        print("the memory size is longer than max_seq_len")
                        sample[key][time_step, flag, :] = torch.from_numpy(self.memory[index][key][time_step])
                    elif ep_len == self.max_seq_len:
                        sample[key][time_step, flag, :] = torch.from_numpy(self.memory[index][key][time_step])
                    else:
                        sample[key][time_step, flag, :] = torch.from_numpy(np.append(self.memory[index][key][time_step],
                                                                    np.zeros(self.max_seq_len - ep_len)))

        return sample

    def sample_sequence(self, batch_size=None, sequence_len=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if sequence_len is not None:
            self.batch_size = sequence_len
        sample = {"s": torch.empty((sequence_len, batch_size, len(self.memory[0]['s'])), dtype=torch.float32),
                  "a": torch.empty((sequence_len, batch_size, len(self.memory[0]['a'])), dtype=torch.float32),
                  "s_": torch.empty((sequence_len, batch_size, len(self.memory[0]['s'])), dtype=torch.float32),
                  "r": torch.empty((sequence_len, batch_size, 1), dtype=torch.float32),
                  "tr": torch.empty((sequence_len, batch_size, 1), dtype=torch.float32)}
        for loop in range(batch_size):
            index = random.sample(range(self.max_position), 1)
            ep_len = len(self.memory[index]['s'])
            for key in self.memory.keys():
                if ep_len <= sequence_len:
                    for time_step in range(sequence_len):
                        sample[key][time_step, loop, :] = torch.from_numpy(np.append(self.memory[index][key][time_step],
                                                                      np.zeros(sequence_len - ep_len)))
                else:
                    start_ = random.sample(range(0, ep_len - sequence_len), 1)
                    end_ = start_ + sequence_len
                    for (time_step, time) in enumerate(range(start_,end_)):
                        sample[key][time_step, loop, :] = torch.from_numpy(self.memory[index][key][time])
        return sample

    def recent_ep_sample(self):
        return self.memory[self.position]

    def __len__(self):
        return len(self.memory)