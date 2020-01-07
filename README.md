# Torch-rl
## introduction
Torch和tensorflow是目前深度学习的主要两个框架，现如今 在 TF 和 torch两个方面都有非常出色的代码，但是从使用程度上来讲torch这边的RL实现，很少有一个兼顾框架和易用的代码。 
这里借鉴了Keras-RL的框架 以及 baseline的实现思路，我尝试建立一套基于Torch版本的RL实现。

本着以最简单的 最快速的 最实际的方式建立一个Torch DRL的框架，节省大家学习的时间直接利用。

希望大家也能加入，一起实现。
## 仓库架构
+ agent 
  包含agent（内含与环境交互的过程） 以及 所有算法
+ common
  包含记录文件、loss函数、经验池、DQN策略
+ model
  包含所有深度网络实现
+ savedata
  记录训练结果
run_xxxxxxxxxxx 训练实例。

## 目前进展
基于Keras-RL建立交互以及算法框架，并借鉴了baseline的logger文件，可以直接输出 txt\CSV\tensorboard对训练过程进行观察
### 架构方面
整体交互代码已经完成。 包含模仿学习过程。
### 算法方面
###### DQN
包含DDQN、Dueling DQN。
### 网络方面
建立全联接网络。


