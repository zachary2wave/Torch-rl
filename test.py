#%%
from torch import nn
import torch
#creat a simple model
model = nn.Sequential(
   nn.Conv3d(1,16,kernel_size=1),
   nn.Conv3d(16,2,kernel_size=1))#tend to print the W of this layer
input = torch.randn([1,1,16,256,256])
if torch.cuda.is_available():
   print('cuda is avaliable')
   model.cuda()
   input = input.cuda()
#%%
#打印某一层的参数名
# for name in model.state_dict():
#    print(name)
#Then  I konw that the name of target layer is '1.weight'
#
# #schemem1(recommended)
# print(model.state_dict()['1.weight'])
#
# #scheme2
# params = list(model.named_parameters())#get the index by debuging
# print(params[2][0])#name
# print(params[2][1].data)#data
#
# #scheme3
# params = {}#change the tpye of 'generator' into dict
# for name,param in model.named_parameters():
# params[name] = param.detach().cpu().numpy()
# print(params['0.weight'])

#scheme4
# for layer in model.modules():
#     if(isinstance(layer,nn.Conv3d)):
#        print(layer.weight)

#打印每一层的参数名和参数值
#schemem1(recommended)
for name, param in model.named_parameters():
   print(name,param)

#scheme2
# for name in model.state_dict():
#    print(name)
#    print(model.state_dict()[name])
