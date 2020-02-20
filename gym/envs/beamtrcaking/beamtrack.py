#%%
from gym import spaces
import numpy as np

from envs.beamtrcaking.channel_RT.multi_resolution_codebook import pcodebook


class beamtrack():

    def __init__(self):

        n = 128
        s = 128
        self.lens = 36 * 2
        self.codebook = pcodebook(n, s, bshow=None)


        wall1 = [[0, 0], [10, 0], [10, 10], [0, 10]]
        wall2 = [[5, 20], [20, 20], [20, 30], [5, 30]]
        wall3 = [[-5, 18], [4, 18], [4, 25], [-5, 25]]
        wall4 = [[15, -5], [50, -5], [50, 5], [15, 5]]
        wall5 = [[25, 17], [45, 17], [45, 27], [25, 27]]
        self.walls = np.array([wall1, wall2, wall3, wall4, wall5])

        self.Tx_xy = np.array([15, 15])  # 设定发射端和接收端的位置信息

        "只考虑单级码本" \
        "动作空间 128个 波束"
        self.actionspace = spaces.Discrete(low=np.array([0]), \
                                       high=np.array([s]), \
                                       dtype=np.float32)
        "状态空间" \
        ""
        self.observationspace = spaces.Box(low=np.array([]), \
                                       high=np.array([self.amax, self.amax]), \
                                       dtype=np.float32)
        # def reset(self):
        #     self.time = 0
        #     index = np.array(range(self.lens))
        #     R = 45
        #     self.Rx = self.Tx_xy[0] + \
        #               R * np.cos(index * 360 / self.lens * math.pi / 180) + \
        #               np.random.normal(0, 2, self.lens)
        #     self.Ry = self.Tx_xy[1] + \
        #               R * np.sin(index * 360 / self.lens * math.pi / 180) + \
        #               np.random.normal(0, 2, self.lens)


        # def step(self,a):
        #     self.time += 1
        #     d_Rx = [self.Rx[(i + 1) % self.lens] - self.Rx[i] for i in range(self.lens)]
        #     d_Ry = [self.Ry[(i + 1) % self.lens] - self.Ry[i] for i in range(self.lens)]
        #     beamFind(self.Rx[self.time], self.Ry[self.time], self.Tx_xy,
        #              self.walls, wallx, wally, vxy[0, self.time], vxy[1, self.time], t)



if __name__ == '__main__':

    env = beamtrack()


