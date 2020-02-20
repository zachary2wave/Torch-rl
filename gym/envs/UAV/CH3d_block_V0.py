import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


class CH3d_block_V0(gym.Env):
    def __init__(self):
        self.NUAV = 1
        self.NSP = 3

        # 约束
        self.Vmax = 20                                 # 最大速度    in m/s
        self.Vmaxz = 10
        self.amax = 5                                  # 最大加速度  in m^2/s
        self.amaxz = 5
        self.MAXp, self.MAXpzlow, self.MAXpzhigh = 400, 40, 100

        self.delta = 0.5                               # 时隙
        self.Pmax = 10                                 #  dBm  功率
        self.choose = 'Urban'
        self.SNRlimit = 0
        self.alpha = 4
        self.done = 0
        # 环境信道参数
        self.B = 1e6                                   # 带宽 1Mhz
        self.N0 = -130                                 # dBm
        self.m = 1900                                  # in g
        self.R_th = 1.3*self.B                         #

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)


        '''
        block 区域 其中srart  为 左下点坐标 
        w ，l， h 分别为 x y z轴 方向的宽度
        '''
        self.blockstartx = [50, -100, -200]
        self.blockstarty = [150, -50, 0]
        self.blockw = [100, 80, 90]
        self.blockl = [70, 90, 60]
        self.blockh = [70, 60, 40]
        self.block = []

        for time in range(len(self.blockstarty)):
            point = [
                [self.blockstartx[time],self.blockstarty[time], 0],
                [self.blockstartx[time]+self.blockw[time], self.blockstarty[time], 0],
                [self.blockstartx[time], self.blockstarty[time]+self.blockl[time], 0],
                [self.blockstartx[time]+self.blockw[time], self.blockstarty[time] + self.blockl[time], 0],
                [self.blockstartx[time], self.blockstarty[time], self.blockh[time]],
                [self.blockstartx[time] + self.blockw[time], self.blockstarty[time], self.blockh[time]],
                [self.blockstartx[time], self.blockstarty[time] + self.blockl[time], self.blockh[time]],
                [self.blockstartx[time] + self.blockw[time], self.blockstarty[time] + self.blockl[time], self.blockh[time]]
                     ]
            self.block.append(point)

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0, 0])                            # 加速度
        self.v = np.array([10, 10, 0])                          # 速度
        self.place = [0, 0, 50]       # 无人机位置x
        # 用户初始化
        self.SPplacex = np.array([])
        self.SPplacey = np.array([])
        for i in range(self.NSP):
            inarea = 1
            while inarea:
                SPplacex = np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置x
                SPplacey = np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置y
                inarea = self.inblock(SPplacex, SPplacey)
            self.SPplacex = np.append(self.SPplacex, SPplacex)
            self.SPplacey = np.append(self.SPplacey, SPplacey)
        self.G0 = np.random.uniform(100, 300, self.NSP)            # 每个节点的数据量 M为单位
        self.P = 10  # 初始发射功率dBm
        self.P_data = 5  # 处理功率  单位W
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1

        # 定义动作
        self.action_space = spaces.Box(low=np.array([-5, -5, -3]),\
                                       high=np.array([5, 5, 3]),\
                                       dtype=np.float32)
        #定义状态空间
        "利用了用户的信噪比 和 数据量 + 无人机位置 x,y,z  "
        self.observation_space = spaces.Box(low=-self.MAXp, high=self.MAXp, shape=(2 * self.NSP + 3,), dtype=np.float32)
        # 定义输出的变量
        self.show = 0
        self.boundary = 1
        self.eps = 0.001

    def reset(self):
            " 初始化无人机 "
            self.a = np.array([0, 0, 0])  # 加速度
            self.v = np.array([10, 10, 0])  # 速度
            self.place = [0, 0, 50]  # 无人机位置
            "重置节点"
            self.SPplacex = np.array([])
            self.SPplacey = np.array([])
            for i in range(self.NSP):
                inarea = 1
                while inarea:
                    SPplacex = np.random.randint(-self.MAXp, self.MAXp, 1)  # 节点位置x
                    SPplacey = np.random.randint(-self.MAXp, self.MAXp, 1)  # 节点位置y
                    inarea = self.inblock(SPplacex, SPplacey)
                self.SPplacex = np.append(self.SPplacex, SPplacex)
                self.SPplacey = np.append(self.SPplacey, SPplacey)
            self.G0 = np.random.uniform(100, 300, self.NSP)
            self.G = np.copy(self.G0)
            self.rate, self.SNR = self.Rate()
            self.cline = np.argmax(self.SNR)
            if self.SNR[self.cline] <= self.SNRlimit:
                self.cline = -1
            # 强化学习部分重置
            self.time = 0
            self.totalR = self.rate[self.cline] * self.delta / 1e6


            # 输出状态
            "利用了用户的信噪比 和 数据量 + 无人机位置 x,y,z 和 无人机速度 vx,vy,vz "
            temp = np.concatenate(([g / 300 for g in self.G], [snr / 10 for snr in self.SNR]), axis=0)
            S = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
            S = np.append(S, [self.place[0] / self.MAXp,
                              self.place[1] / self.MAXp,
                              self.place[2] / (self.MAXpzhigh-self.MAXpzlow)])
            return S

    def step(self, a):
        reward = 0
        done = 0
        self.a = a
        self.P = self.Pmax
        self.time += 1

        x, y, z = [self.place[i] + a[i] for i in range(len(a))]
        tp1 = self.bound(x, self.MAXp, -self.MAXp)
        tp2 = self.bound(y, self.MAXp, -self.MAXp)
        tp3 = self.bound(z, self.MAXpzhigh, self.MAXpzlow)
        tp4 = self.inblock(x, y, z)
        if tp4:
            reward += -(tp1 + tp2 + tp3 + tp4) * 10
        else:
            self.place = [(1 - tp1) * x + tp1 * self.place[0],
                          (1 - tp2) * y + tp2 * self.place[1],
                          (1 - tp3) * x + tp3 * self.place[2]]
            # 判断所链接的用户
            self.rate, self.SNR = self.Rate()
            self.cline = np.argmax(self.rate)
            # 判断信噪比
            output = 0
            if self.SNR[self.cline] <= self.SNRlimit:
                self.cline = -1
            else:
                Gidea = self.rate[self.cline] * self.delta / 1e6
                if Gidea < self.G[self.cline]:
                    output = Gidea
                    self.G[self.cline] -= output
                else:
                    output = self.G[self.cline]
                    self.G[self.cline] = 0
            '''
            the second reward
            '''
            # if output > 0:
            #     reward = 1
            # else:
            #     reward = -1
            '''
            the first reward
            '''
            if output > 0:
                reward += output * self.delta * 10
            else:
                reward += -1
        teskleft = np.sum(self.G)
        if teskleft == 0 or self.time == 10000:
            done = 1

        temp = np.concatenate(([g / 300 for g in self.G], [snr / 10 for snr in self.SNR]), axis=0)
        S_ = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S_ = np.append(S_, [self.place[0] / self.MAXp,
                          self.place[1] / self.MAXp,
                          self.place[2] / self.MAXphighz])
        return S_, reward / 100, done, {'G0': np.sum(self.G0)}

    def inblock(self, x, y, z):
        blocknum = len(self.blockstartx)
        flag = 0
        for time in range(blocknum):
            if self.blockstartx[time] < x < self.blockstartx[time] + self.blockw[time] \
                    and self.blockstarty[time] < y < self.blockstarty[time] + self.blockl[time]\
                    and z < self.blockh[time]:
                flag = 1
                break
        return flag

    # 判断速度是否在可行范围内
    def bound(self, v, upband, lowband):
        flag = int(v > upband) + int(v < lowband)
        return flag

    def Rate(self):
        inta_los, inta_Nlos = 0, 0
        if self.choose == 'subruban':
            inta_los = 0.1
            inta_Nlos = 21
        elif self.choose == 'Urban':
            inta_los = 1
            inta_Nlos = 20
        elif self.choose == 'DenseUrban':
            inta_los = 1.6
            inta_Nlos = 23
        elif self.choose == 'HighUrban':
            inta_los = 2.3
            inta_Nlos = 34

        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[i]) ** 2 +
                          (self.placey - self.SPplacey[i]) ** 2 +
                          (self.placez) ** 2)

            flag = 0
            for bl in range(len(self.blockl)):
                # xoy面
                block = self.block[bl]
                K = (self.placey - self.SPplacey[i])/(self.placex - self.SPplacex[i] + self.eps)
                dot1 = np.sign(K * (block[0][0] - self.placex) - (block[0][1] - self.placey))
                dot2 = np.sign(K * (block[1][0] - self.placex) - (block[1][1] - self.placey))
                dot3 = np.sign(K * (block[2][0] - self.placex) - (block[2][1] - self.placey))
                dot4 = np.sign(K * (block[3][0] - self.placex) - (block[3][1] - self.placey))
                if dot1>0 and dot2>0 and dot3>0 and dot4>0:
                    flag += 1
                elif dot1<0 and dot2<0 and dot3<0 and dot4<0:
                    flag += 1

                # xoz面
                K = (self.placez)/(self.placex - self.SPplacex[i] + self.eps)
                dot1 = np.sign(K * (block[0][0] - self.placex) - (block[0][1] - self.placez))
                dot2 = np.sign(K * (block[1][0] - self.placex) - (block[1][1] - self.placez))
                dot3 = np.sign(K * (block[4][0] - self.placex) - (block[4][1] - self.placez))
                dot4 = np.sign(K * (block[5][0] - self.placex) - (block[5][1] - self.placez))
                if dot1>0 and dot2>0 and dot3>0 and dot4>0:
                    flag += 1
                elif dot1<0 and dot2<0 and dot3<0 and dot4<0:
                    flag += 1

                # yoz面
                K = (self.placez)/(self.placey - self.SPplacey[i] + self.eps)
                dot1 = np.sign(K * (block[0][1] - self.placey) - (block[0][1] - self.placez))
                dot2 = np.sign(K * (block[2][1] - self.placey) - (block[2][1] - self.placez))
                dot3 = np.sign(K * (block[4][1] - self.placey) - (block[4][1] - self.placez))
                dot4 = np.sign(K * (block[6][1] - self.placey) - (block[6][1] - self.placez))
                if dot1 > 0 and dot2 > 0 and dot3 > 0 and dot4 > 0:
                    flag += 1
                elif dot1 < 0 and dot2 < 0 and dot3 < 0 and dot4 < 0:
                    flag += 1

            PLmax = (10 * math.log10(D ** self.alpha) + self.lossb + (1 - int(flag == 6)) * inta_los + int(flag == 6) * inta_Nlos)
            SNR[i] = self.P - PLmax - self.N0
            rate[i] = self.B * math.log2(1 + self.IdB(SNR[i]))
        return rate, SNR

    def listdo(self,a,b,label):
        if label == 1:
            return [a[i]+b[i] for i in range(len(b))]
        elif label == 2:
            return [a[i]-b[i] for i in range(len(b))]
        elif label == 3:
            return [a[i]*b[i] for i in range(len(b))]
        elif label == 4:
            return [a[i]/b[i] for i in range(len(b))]
        elif label == 5:
            return [a*b[i] for i in range(len(b))]

        # 计算瞬时时间延迟 loss