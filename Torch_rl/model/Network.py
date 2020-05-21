import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Normal, Categorical
from torch.autograd import Variable
from copy import deepcopy
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

        self.gpu = False

    def forward(self, x):
        if self.gpu:
            x = x.cuda(self.device)
        for layer in self.linears:
            x = layer(x)
        return x

    @property
    def layer_infor(self):
        return list(self._layer_num)

    def to_gpu(self, device=None):
        self.linears.cuda(device=device)
        self.gpu = True
        self.device = device


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
        self.gpu = None


    def init_H_C(self, batch_size):
        if self.gpu:
            return (Variable(torch.zeros(1, batch_size, self.lstm_unit)).cuda(),
                    Variable(torch.zeros(1, batch_size, self.lstm_unit)).cuda())
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_unit)),
                    Variable(torch.zeros(1, batch_size, self.lstm_unit)))

    def forward(self, x, h_state=None):
        if self.gpu is not None:
            x = x.cuda(self.gpu)
        if h_state is None:
            h_state = self.init_H_C(x.size()[1])
        x, h_state = self.LSTM(x, h_state)
        x = self.hidden_activate(x)
        x = self.Dense(x)
        return x, h_state

    def to_gpu(self, device=None):
        self.LSTM.cuda(device=device)
        self.Dense.cuda(device=device)
        self.gpu = True
        self.device = device


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

        first = [input_size[0]]+[kernal[0] for kernal in kernal_size ]

        cnnlayer=[]
        for flag, kernal in enumerate(kernal_size):
            cnnlayer.append(("cnn" + str(flag), nn.Conv2d(first[flag], kernal[0], kernel_size=kernal[1],
                                                 stride=stride, padding=padding, padding_mode=padding_mode)))
            cnnlayer.append(("cnn_activate" + str(flag), deepcopy(hidden_activate)))
            if pooling_way == "Max":
                cnnlayer.append(("pooling" + str(flag),torch.nn.MaxPool2d(kernel_size=pooling_kernal, stride=pooling_stride)))
            elif pooling_way == "Ave":
                cnnlayer.append(("pooling" + str(flag),torch.nn.AvgPool2d(kernel_size=pooling_kernal, stride=pooling_stride)))

        self.CNN = nn.Sequential(OrderedDict(cnnlayer))
        self.input_dense = self.size_cal(input_size)
        self.Dendse = DenseNet(self.input_dense, output_size, hidden_layer=dense_layer,
                               hidden_activate=hidden_activate, output_activate=output_activate,BatchNorm=BatchNorm)
        self.gpu = False

    def size_cal(self, input_size):
        test_input = torch.rand((1,)+input_size)
        test_out = self.CNN(test_input)
        return test_out.size(1)*test_out.size(2)*test_out.size(3)

    def forward(self, x):
        if self.gpu:
            x = x.cuda(self.device)
        x = self.CNN(x)
        x = x.view(x.size(0), -1)
        x = self.Dendse(x)
        return x

    def to_gpu(self, device=None):
        self.CNN.cuda(device=device)
        self.Dense.cuda(device=device)
        self.gpu = True
        self.device = device

class CNN_2D_LSTM_Dense(nn.Module):
    def __init__(self, input_size, output_size,
                 # CNN_layer
                 kernal_size=[(32, 7), (64, 5), (128, 3)],
                 stride=1, padding=0,  padding_mode='zeros',
                 # pooling
                 pooling_way = "Max", pooling_kernal= 2, pooling_stride = 2,
                 # LSTM
                 lstm_unit=64, lstm_layer=1,
                 # Dense
                 dense_layer = [64, 64], hidden_activate=nn.ReLU(), output_activate=None,
                 BatchNorm = False):
        super(CNN_2D_LSTM_Dense, self).__init__()

        first = [input_size[0]]+[kernal[0] for kernal in kernal_size ]
        if pooling_way == "Max":
            poollayer = torch.nn.MaxPool2d(kernel_size=pooling_kernal, stride=pooling_stride)
        elif pooling_way == "Max":
            poollayer = torch.nn.AvgPool2d(kernel_size=pooling_kernal, stride=pooling_stride)
        cnnlayer=[]
        for flag, kernal in enumerate(kernal_size):
            cnnlayer.append(("cnn" + str(flag), nn.Conv2d(first[flag], kernal[0], kernel_size=kernal[1],
                                                 stride=stride,padding=padding,padding_mode=padding_mode)))
            cnnlayer.append(("cnn_activate" + str(flag), deepcopy(hidden_activate)))
            cnnlayer.append(("pooling" + str(flag), deepcopy(poollayer)))

        self.CNN = nn.Sequential(OrderedDict(cnnlayer))
        self.input_lstm = self.size_cal(input_size)
        self.lstm_unit = lstm_unit
        self.LSTM = nn.LSTM(input_size=self.input_lstm,
                            hidden_size=lstm_unit,
                            num_layers=lstm_layer)

        self.Dendse = DenseNet(lstm_unit, output_size, hidden_layer=dense_layer,
                               hidden_activate=hidden_activate, output_activate=output_activate,BatchNorm=BatchNorm)

        self.gpu = False

    def init_H_C(self, batch_size):
        if self.gpu:
            return (Variable(torch.zeros(1, batch_size, self.lstm_unit)).cuda(self.device),
                    Variable(torch.zeros(1, batch_size, self.lstm_unit)).cuda(self.device))
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_unit)),
                    Variable(torch.zeros(1, batch_size, self.lstm_unit)))

    def size_cal(self, input_size):
        test_input = torch.rand((1,)+input_size)
        test_out = self.CNN(test_input)
        return test_out.size(1)*test_out.size(2)*test_out.size(3)

    def forward(self, x, h = None):
        if self.gpu:
            x = x.cuda(self.device)
        batch_size = x.size(1)
        squence_size = x.size(0)
        conjection = ()
        for time in range(squence_size):
            cnnout = self.CNN(x[time])
            cnnout = cnnout.view(batch_size, -1)
            cnnout = cnnout.unsqueeze(0)
            conjection = conjection + (cnnout,)
        x = torch.cat(conjection, dim=0)
        if h is None:
            h = self.init_H_C(batch_size)
        x, h = self.LSTM(x, h)
        x = self.Dendse(x)
        return x

    def to_gpu(self, device=None):
        self.CNN.cuda(device=device)
        self.LSTM.cuda(device=device)
        self.Dense.cuda(device=device)
        self.gpu = True
        self.device = device

class LSTM_Dense_Hin(nn.Module):
    def __init__(self, input_size, output_size, lstm_unit=64, lstm_layer=1, dense_layer=[64, 64],
                 hidden_activate=nn.ReLU(), output_activate=None,
                 BatchNorm=False):
        super(LSTM_Dense_Hin, self).__init__()
        self.lstm_unit = lstm_unit
        self.hidden_activate = nn.ReLU()
        self.Dense = DenseNet(lstm_unit, output_size, hidden_layer=dense_layer,
                              hidden_activate=hidden_activate, output_activate=output_activate, BatchNorm=BatchNorm)

        self._layer_num = [input_size] + [lstm_unit] * lstm_layer + dense_layer + [output_size]
        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_unit,
                            num_layers=lstm_layer)
        self.h_state = None
        self.gpu = False

    def init_H_C(self, batch_size):
        if self.gpu:
            return (Variable(torch.normal(mean=torch.zeros(1, batch_size, self.lstm_unit)).cuda(self.device)),
                    Variable(torch.normal(mean=torch.zeros(1, batch_size, self.lstm_unit)).cuda(self.device)))
        else:
            return (Variable(torch.normal(mean=torch.zeros(1, batch_size, self.lstm_unit))),
                    Variable(torch.normal(mean=torch.zeros(1, batch_size, self.lstm_unit))))

    def forward(self, x):
        if self.gpu:
            x = x.cuda(self.device)
        if len(x.shape) == 1:
            # forward
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            # backward
            x = x.unsqueeze(1)
        if self.h_state is None:
            self.h_state = self.init_H_C(1)
        x, self.h_state = self.LSTM(x, self.h_state)
        x = self.hidden_activate(x)
        x = self.Dense(x)
        return x[0]

    def reset_h(self):
        self.h_state = None

    def to_gpu(self, device=None):
        self.LSTM.cuda(device=device)
        self.Dense.to_gpu(device=device)
        self.gpu = True
        self.device = device

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.gpu = False

    def forward(self, input_tensor, hidden_state=None):
        if self.gpu:
            input_tensor = input_tensor.cuda(self.device)
            hidden_state = hidden_state.cuda(self.device)
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def to_gpu(self, device=None):
        self.cell_list.cuda(device=device)
        self.gpu = True
        self.device = device


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        from Torch_rl.model.gcn_layers import GraphConvolution
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)