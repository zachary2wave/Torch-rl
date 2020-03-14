import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.distributions import Normal, Categorical
from torch.autograd import Variable
# from graphviz import Digraph

#
# def make_dot(var, params=None):
#     """ Produces Graphviz representation of PyTorch autograd graph
#     Blue nodes are the Variables that require grad, orange are Tensors
#     saved for backward in torch.autograd.Function
#     Args:
#         var: output Variable
#         params: dict of (name, Variable) to add names to node that
#             require grad (TODO: make optional)
#     """
#     if params is not None:
#         assert isinstance(params.values()[0], Variable)
#         param_map = {id(v): k for k, v in params.items()}
#
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()
#
#     def size_to_str(size):
#         return '(' + (', ').join(['%d' % v for v in size]) + ')'
#
#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 name = param_map[id(u)] if params is not None else ''
#                 node_name = '%s\n %s' % (name, size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#
#     add_nodes(var.grad_fn)
#     return dot
#
# def show(net):
#     x = Variable(torch.randn(net.layer_infor[0]))
#     y = net(x)
#     g = make_dot(y)
#     # g.view()
#     return g


class DenseNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer=[64, 64],
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        super(DenseNet, self).__init__()
        first_placeholder = np.insert(np.array(hidden_layer), 0, input_size)
        second_placeholder = np.array(hidden_layer)
        self._layer_num = np.append(first_placeholder, output_size)
        self.layer = []
        for i in range(len(second_placeholder)):
            layer = []
            layer.append(('linear'+str(i), nn.Linear(first_placeholder[i], second_placeholder[i], bias=True)))
            layer.append(('activation'+str(i),hidden_activate))
            if BatchNorm:
                layer.append(('BatchNormalization'+str(i), nn.BatchNorm1d(second_placeholder[i], eps=1e-05, momentum=0.1, affine=True,
                                                 track_running_stats=True)))
            self.layer.append(nn.Sequential(OrderedDict(layer)))
        output_layerlayer = [("output_layer",nn.Linear(first_placeholder[-1], output_size, bias=True))]
        if output_activate is not None:
            output_layerlayer.append(("output_activation",output_activate))
        self.layer.append(nn.Sequential(OrderedDict(output_layerlayer)))
        self.linears = nn.ModuleList(self.layer)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x

    @property
    def layer_infor(self):
        return list(self._layer_num)


class LSTM_Dense(nn.Module):
    def __init__(self, input_size, output_size, lstm_unit=64, lstm_layer=1, dense_layer=[64, 64],
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        super(LSTM_Dense, self).__init__()
        self.lstm_unit = lstm_unit
        self.hidden_activate = nn.ReLU()
        self.Dense = DenseNet(lstm_unit, output_size, hidden_layer=dense_layer,
                              hidden_activate=hidden_activate, output_activate=output_activate, BatchNorm=BatchNorm)

        self._layer_num = [input_size] + [lstm_unit] * lstm_layer + dense_layer + [output_size]
        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_unit,
                            num_layers=lstm_layer)

    def init_H_C(self,batch_size):
        return (Variable(torch.zeros(1, batch_size, self.lstm_unit)),
                Variable(torch.zeros(1, batch_size, self.lstm_unit)))

    def forward(self, x, h_state):
        if h_state is None:
            h_state = self.init_H_C(x.size()[1])
        x, h_state = self.LSTM(x, h_state)
        x = self.hidden_activate(x)
        x = self.Dense(x)
        return x, h_state


class CNN_2D_Dense(nn.Module):
    def __init__(self, input_size, output_size,
                 # CNN_layer
                 kernal_size=[(32, 7), (64, 5), (128, 3)],
                 stride=1, padding=0,  padding_mode='zeros',
                 # pooling
                 pooling_way = "Max", pooling_kernal= 2, pooling_stride = 2,
                 # Dense
                 dense_layer = [64, 64], hidden_activate=nn.ReLU(), output_activate=None,
                 BatchNorm = False):
        super(CNN_2D_Dense, self).__init__()

        first = [input_size]+[kernal[0] for kernal in kernal_size ]
        if pooling_way == "Max":
            poollayer = torch.nn.MaxPool2d(kernel_size=pooling_kernal, stride=pooling_stride)
        elif pooling_way == "Max":
            poollayer = torch.nn.AvgPool2d(kernel_size=pooling_kernal, stride=pooling_stride)
        cnnlayer=[]
        for flag, kernal in enumerate(kernal_size):
            cnnlayer.append(("cnn" + str(flag), nn.Conv2d(first, kernal[0], kernel_size=kernal[1],
                                                 stride=stride,padding=padding,padding_mode=padding_mode)))
            cnnlayer.append(("cnn_activate" + str(flag), deepcopy(hidden_activate)))
            cnnlayer.append(("pooling" + str(flag), deepcopy(poollayer)))

        self.CNN = nn.Sequential(OrderedDict(cnnlayer))
        self.input_dense = kernal_size[-1][0]*kernal_size[-1][1]*kernal_size[-1][1]
        self.Dendse = DenseNet(self.input_dense, output_size, hidden_layer=dense_layer,
                               hidden_activate=hidden_activate, output_activate=output_activate,BatchNorm=BatchNorm)

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(x.size(0), -1)
        x = self.Dendse(x)
        return x