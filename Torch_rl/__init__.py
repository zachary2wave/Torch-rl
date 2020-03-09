from .common import logger
#  agent
from Torch_rl.agent.DQN import DQN_Agent as DQN
from Torch_rl.agent.DDPG import DDPG_Agent as DDPG
from Torch_rl.agent.PPO import PPO_Agent as Batch_PPO
from Torch_rl.agent.PPO3 import PPO_Agent as PPO
from Torch_rl.agent.TD3 import TD3_Agent as TD3
from Torch_rl.agent.HIRO import HIRO_Agent as HIRO

#  network
from Torch_rl.model.Network import DenseNet



