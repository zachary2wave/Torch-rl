# Torch-rl
## introduction
Torch和tensorflow是目前深度学习的主要两个框架，现如今 在 TF 和 torch两个方面都有非常出色的代码，但是从使用程度上来讲torch这边的RL实现，很少有一个兼顾框架和易用的代码。 
这里借鉴了Keras-RL的框架 以及 baseline的实现思路，建立一套基于Torch版本的RL实现。

本着以最简单的 最快速的 最实际的方式建立一个Torch DRL的框架，节省大家学习的时间直接利用。希望大家也能加入，一起实现。

**本仓库兼容CPU与GPU，目前还未实现MPI。** 算法（非严格）采用PET - 8编写， 并带有注释。

## 仓库架构
+ agent 
  包含agent 内核（与环境交互的过程） 以及 所有强化学习算法
+ common
  包含记录文件、loss函数、经验池、DQN策略
+ model
  包含所有深度网络实现
+ savedata
  记录训练结果
run_xxxxxxxxxxx 训练实例。

## 目前进展
基于Keras-RL建立交互以及算法框架，并借鉴了baseline的logger文件，可以直接输出 txt、CSV、tensorboard对训练过程进行观察

#### 算法：

+ DQN（包含Double DQN、 Dueling DQN）[source code](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/DQN.py)

+ DRQN [source code](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/DRQN.py)

+ DDPG [source code](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/DDPG.py)

+ PPO    [source code ](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/PPO3.py)

+ Batch-PPO [source code ](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/PPO.py)

+ TD3 [source code ](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/agent/TD3.py)

#### 网络:

可以快速建立全联接网络、CNN、LSTM、CNN-LSTM。





## example

有一些简单的训练example

[RUN_Catrpole_with_DQN.py](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/RUN_Catrpole_with_DQN.py)

[RUN_Pendulum_with_DDPG.py](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/RUN_Pendulum_with_DDPG.py)

[RUN_Pendulum_with_PPO.py](https://github.com/zachary2wave/Torch-rl/blob/master/Torch_rl/RUN_Pendulum_with_PPO.py)

## 教程 等待进一步更新。。。。。。


