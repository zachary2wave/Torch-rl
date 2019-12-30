import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.distributions import Normal, Categorical



class DenseNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer=np.array([64,64]),
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        super(DenseNet, self).__init__()
        first_placeholder = np.insert(hidden_layer, 0, input_size)
        second_placeholder = hidden_layer
        self._layer_num = np.append(first_placeholder,output_size)
        layer = []

        for i in range(len(second_placeholder)):
            layer.append(('layer'+str(i), nn.Linear(first_placeholder[i], second_placeholder[i],bias=True)))
            layer.append(('hidden_activation' + str(i), hidden_activate))
            if BatchNorm:
                nn.BatchNorm1d(second_placeholder[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        layer.append(('output_layer', nn.Linear(first_placeholder[-1], output_size)))
        if output_activate is not None:
            layer.append(('output_activation', output_activate))
        self.model = nn.Sequential(OrderedDict(layer))

    def forward(self, obs):
        return self.model(obs)

    @property
    def layer_infor(self):
        return list(self._layer_num)



