#%%
import torch
from model.Network import DenseNet
import numpy as np
from torch import nn
import random
from torch.autograd import Variable

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        self.linears = nn.ModuleList([self.layer1, self.layer2])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for l in self.linears:
            x = l(x)
        return x

input = torch.randn(100, 10)
net = DenseNet(10, 2)
output = net.forward(input)
from torch.utils.tensorboard import SummaryWriter
with SummaryWriter('./outcome/') as writter:
    writter.add_graph(net, input)

