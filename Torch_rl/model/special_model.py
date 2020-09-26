
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Normal, Categorical
from torch.autograd import Variable
from copy import deepcopy


class Multi_in(nn.Module):
    def __init__(self, observation_size, action_size, hidden_up_layer=[64, 64], hidden_down_layer=[64, 64],
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        super(Multi_in, self).__init__()

        self.up_layer1 = nn.Linear(observation_size, hidden_up_layer[0], bias=True)
        self.up_layer2 = nn.Linear(hidden_up_layer[0], hidden_up_layer[1], bias=True)
        self.down_layer1 = nn.Linear(hidden_up_layer[1]*3, hidden_down_layer[0], bias=True)
        self.down_layer2 = nn.Linear(hidden_down_layer[0], hidden_up_layer[1], bias=True)
        self.outpu_layer3 = nn.Linear(hidden_down_layer[1], action_size+1, bias=True)

        self.hidden_activate = hidden_activate
        self.output_activate = output_activate

        self.gpu = False

    def forward(self, x1,x2,x3):

        x1 = self.hidden_activate(self.up_layer1(x1))
        x1 = self.hidden_activate(self.up_layer2(x1))

        x2 = self.hidden_activate(self.up_layer1(x2))
        x2 = self.hidden_activate(self.up_layer2(x2))

        x3 = self.hidden_activate(self.up_layer1(x3))
        x3 = self.hidden_activate(self.up_layer2(x3))
        x = torch.cat([x1,x2,x3], dim=-1)
        x = self.hidden_activate(self.down_layer1(x))
        x = self.hidden_activate(self.down_layer2(x))
        x = self.hidden_activate(self.output_layer1(x))
        Q = x[0]+torch.mean(x[1:])
        return Q

