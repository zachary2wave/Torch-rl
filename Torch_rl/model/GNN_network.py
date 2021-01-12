
import math
import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix



" The GCN Part"
def normalize(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx

def normalize_sparse(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class GraphConvolution(Module):
    def __init__(self, adj, in_features, out_features,
                 activate=nn.ReLU(), sparse_inputs=False, chebyshev_polynomials = 0, dropout = 0.5, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activate = activate
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


        if sparse_inputs:
            adj = adj.toarray()
        if chebyshev_polynomials>0:
            T_K = []
            T_K.append(np.eye(adj.shape[0]))
            laplacian = np.eye(adj.shape[0]) - normalize(adj)
            largest_eigval, _ = np.linalg.eig(laplacian)
            scaled_laplacian = (2. / largest_eigval[0]) * laplacian - np.eye(adj.shape[0])
            T_K.append(scaled_laplacian)
            for i in range(2, chebyshev_polynomials+1):
                T_K.append(2 * np.dot(scaled_laplacian,T_K[-1])-T_K[-2])
            self.T_k = T_K
        else:
            self.T_k = [normalize(adj)]
        if sparse_inputs:
            self.adj = [coo_matrix(T) for T in self.T_k]
        else:
            self.adj = self.T_k

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        output = torch.zeros_like(support)
        if self.sparse_inputs:
            for adj in self.adj:
                output = output + torch.sparse.mm(adj, support)
        else:
            for adj in self.adj:
                output = output + torch.mm(adj, support)
        if self.bias is not None:
            return self.activate(output + self.bias)
        else:
            return self.activate(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



"The Graph SAGE"




"The Graph Attention Network"
