#%%
import torch
from model.Network import DenseNet
import numpy as np
from torch import nn
import random

model = DenseNet(10, 2, hidden_layer=np.array([5]),
                       hidden_activate=nn.ReLU())

input = np.random.normal(size=(10, 1))
input = input.astype(np.float32)
input = torch.from_numpy(input).unsqueeze(0)

input = torch.randn(100, 10)


output = model.forward(input)

