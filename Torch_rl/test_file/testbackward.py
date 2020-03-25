

from Torch_rl.model.Network import DenseNet
from torch import nn
import torch
from torch.optim import Adam
policy_model = DenseNet(12, 2,
                 hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
input  = torch.rand(size=(32,12))
output = torch.rand(size=(32,2))
loss_cal1 = torch.nn.SmoothL1Loss()
loss_cal2 = torch.nn.MSELoss()
policy_model_optim = Adam(policy_model.parameters(), lr=1e-4)
for time in range(100):
    y = policy_model.forward(input)
    loss = loss_cal1(y, output)
    policy_model_optim.zero_grad()
    loss.backward(retain_graph=True)
    policy_model_optim.step()
    loss = loss_cal2(y, output)
    policy_model_optim.zero_grad()
    loss.backward()
    policy_model_optim.step()