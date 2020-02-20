import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# import panda as pd
import scipy.io as sio
import time as TT
'''
11.2 建立碰撞模型

'''
class HLC2d(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 3
        self.delta = 0.5                             # 时隙
        self.T = 1000                                  # 总时间
        self.time = 0

        self.N = self.T/self.delta
        self.choose = 'Urban'
        self.SNRlimit = 0
        self.alpha = 4
        # 环境信道参数
        self.B = 1e6                                   # 带宽 1Mhz
        self.N0 = -130                                 # dBm
        self.m = 1900                                  # in g
        self.R_th = 1.3*self.B                         #
        self.inta_los, self.inta_Nlos = 0, 0
        if self.choose == 'subruban':
                self.inta_los = 0.1
                self.inta_Nlos = 21
        elif self.choose == 'Urban':
                self.inta_los = 1
                self.inta_Nlos = 20
        elif self.choose == 'DenseUrban':
                self.inta_los = 1.6
                self.inta_Nlos = 23
        elif self.choose == 'HighUrban':
                self.inta_los = 2.3
                self.inta_Nlos = 34

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)
        '''
        block 区域 其中a 为 左上点坐标 b为 右下点坐标 
        也就是ax一定小于bx  ay一定大于by
        '''
        self.blockstartx = [50, -200, -200]
        self.blockstarty = [150, -200, 0]
        self.blockw = [100, 160, 70]
        self.blockl = [140, 80, 170]

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0])                                # 加速度
        self.v = np.array([10, 10])                              # 速度
        self.placex = 0                                          # 无人机位置x
        self.placey = 0                                          # 无人机位置y
        self.placey = 50                                         # 无人机位置z
        self.MAXboundary = 400

        self.SPplacex = np.array([])
        self.SPplacey = np.array([])
        for i in range(self.NSP):
            inarea = 1
            while inarea:
                SPplacex = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置x
                SPplacey = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置y
                inarea = self.inblock(SPplacex, SPplacey)
            self.SPplacex = np.append(self.SPplacex, SPplacex)
            self.SPplacey = np.append(self.SPplacey, SPplacey)
        self.G0 = np.random.uniform(100, 300, self.NSP)            # 每个节点的数据量 M为单位
        self.Grecord = np.copy(self.G0)
        self.P = 10                                              # 初始发射功率dBm
        self.P_data = 5                                          # 处理功率  单位W
        self.PLmax = self.PLoss()
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        # 定义动作空间

        self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)
        "利用了用户的信噪比 和 数据量 + 无人机位置 x,y 和 无人机速度 vx,vy "
        self.observation_space = spaces.Box(low=-self.MAXboundary, high=self.MAXboundary, shape=(2*self.NSP+2,), dtype=np.float32)
        # 定义输出的变量
        self.show = 0
        self.boundary = 1

    def reset(self):
        # 初始化强化学习
        self.done = 0
        self.time = 0
        self.totalR = self.rate[self.cline] * self.delta / 1e6
        # 初始化无人机 和 node节点 数据
        self.a = np.array([0, 0])    # 加速度
        self.v = np.array([10, 10])  # 速度
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
        "重置节点位置"
        self.SPplacex = np.array([])
        self.SPplacey = np.array([])
        for i in range(self.NSP):
            inarea = 1
            while inarea:
                SPplacex = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置x
                SPplacey = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置y
                inarea = self.inblock(SPplacex, SPplacey)
            self.SPplacex = np.append(self.SPplacex, SPplacex)
            self.SPplacey = np.append(self.SPplacey, SPplacey)
        self.G0= np.random.uniform(100, 300, self.NSP)
        self.G = np.copy(self.G0)
        self.time = 0
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.SNR)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        temp = np.concatenate(([g/300 for g in self.G],[snr/10 for snr in self.SNR]), axis=0)
        S = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S = np.append(S, [self.placex/self.MAXboundary, self.placey/self.MAXboundary])

        return S

    def step(self, a):
        reward = 0
        self.done = 0
        self.time +=1
        x = self.placex + a[0]
        y = self.placey + a[1]
        tp1 = self.bound(x, self.MAXboundary, -self.MAXboundary)
        tp2 = self.bound(y, self.MAXboundary, -self.MAXboundary)
        tp4 = self.inblock(x, y)
        if tp4:
            reward += -(tp1 + tp2 + tp4) * 10
        else:
            self.placex = (1 - tp1) * x + tp1 * self.placex   # 无人机位置x
            self.placey = (1 - tp2) * y + tp2 * self.placey   # 无人机位置y
            # 判断所链接的用户
            self.rate, self.SNR = self.Rate()
            self.cline = np.argmax(self.rate)
            # 判断信噪比
            output = 0
            if self.SNR[self.cline] <= self.SNRlimit:
                self.cline = -1
            else:
                Gidea = self.rate[self.cline]*self.delta/1e6
                if Gidea < self.G[self.cline]:
                    output = Gidea
                    self.G[self.cline] -= output
                else:
                    output = self.G[self.cline]
                    self.G[self.cline] = 0
            self.totalR += output
            '''
            the second reward
            '''
            if output > 0:
                reward += output * self.delta * 10
            else:
                reward += -1
            '''
            the first reward
            '''
            # if output > 0:
            #     reward += output*10
            # else:
            #     reward += -teskleft*1e-2
        teskleft = np.sum(self.G)
        if teskleft == 0 or self.time == 10000:
            self.done = 1

        temp = np.concatenate(([g/300 for g in self.G], [snr/10 for snr in self.SNR]), axis=0)
        S_ = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S_ = np.append(S_, [self.placex/self.MAXboundary , self.placey/self.MAXboundary ])
        return S_, reward/100, self.done, {'G0':np.sum(self.G0)}
# 判断生成的点是否在可行范围内
    def inblock(self,x,y):
        blocknum = len(self.blockstartx)
        flag = 0
        for time in range(blocknum):
            if self.blockstartx[time] < x < self.blockstartx[time]+self.blockw[time] \
                    and self.blockstarty[time] < y < self.blockstarty[time]+self.blockl[time]:
                flag = time+1
                break
        return flag

    # 判断是否在可行范围内
    def bound(self, v, upband, lowband):
        flag = int(v > upband) + int(v < lowband)
        return flag

    # inline
    def inline(self, aimplacex, aimplacey):
        K = ((self.placey - aimplacey)+1e-4)/ ((self.placex - aimplacex)+1e-4)
        B = (self.placex - aimplacex)
        flag = 0
        for bl in range(len(self.blockstartx)):
            xmin, xmax = self.blockstartx[bl],self.blockstartx[bl]+self.blockw[bl]
            ymin, ymax = self.blockstarty[bl],self.blockstarty[bl]+self.blockl[bl]
            yinline = K * self.blockstartx[bl]+B
            if ymin<yinline<ymax and min([self.placey,aimplacey])<yinline<max([self.placey,aimplacey]):
                flag = 1
            yinline = K * (self.blockstartx[bl]+self.blockw[bl])+B
            if ymin < yinline < ymax and \
               min([self.placey, aimplacey]) < yinline < max([self.placey, aimplacey]):
                flag = 2
            xinline = (self.blockstarty[bl]-B)/K
            if xmin < xinline < xmax and \
                min([self.placex, aimplacex]) < yinline < max([self.placex, aimplacex]):
                flag = 3
            xinline = (self.blockstarty[bl]+self.blockl[bl]-B)/K
            if xmin < xinline < xmax and \
                min([self.placex, aimplacex]) < yinline < max([self.placex, aimplacex]):
                flag = 4
        return flag

    # 计算瞬时信噪比
    def Rate(self):
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[i]) ** 2 +
                          (self.placey - self.SPplacey[i]) ** 2 +
                          (self.placez) ** 2)
            flag = int(bool(self.inline(self.SPplacex[i], self.SPplacey[i])))
            PLmax = 10 * math.log10(D ** self.alpha) + self.lossb + flag * self.inta_los + (1 - flag) * self.inta_Nlos
            SNR[i] = self.P-PLmax-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR

# 计算顺时功率
    def P_calfly(self):
        C1 = 9.26e-4
        C2 = 2250
        g  = 9.8
        normV = np.linalg.norm(self.v)
        norma = np.linalg.norm(self.a)
        cos = np.sum(self.v*self.a)
        Ps = C1*normV**3+C2/normV*(1+(norma**2-cos**2/normV**2)/(g**2))
        return Ps

    def relative_position(self):
        replacex = self.SPplacex - self.placex
        replacey = self.SPplacey - self.placey
        return replacex, replacey
# 计算dB
    def dB(self,a):
        b = 10*math.log10(a/10)
        return b

    def IdB(self, a):
        b = math.pow(10,a/10)
        return b
# 画图三维
    def drawplot(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.scatter(self.placex, self.placey, 100)
        ax.scatter(self.SPplacex, self.SPplacey, np.zeros_like(self.SPplacex))
        ax.text(self.placex, self.placey, self.placez,
                'loc='+str([self.placex, self.placey, self.placez])+'\n'
                +'V='+str(self.v)+'\n'+'P='+str(self.P))
        if self.cline != -1:
            ax.plot([self.placex, self.SPplacex[self.cline]], [self.placey, self.SPplacey[self.cline]],
                    [self.placez, 0], '--')
            ax.text((self.placex + self.SPplacex[self.cline])/2, (self.placey+self.SPplacey[self.cline])/2,
                    (self.placez + 0)/2, str(self.rate[self.cline]))
            ax.text(self.SPplacex[self.cline], self.SPplacex[self.cline], self.SPplacex[self.cline],
                    'loc='+str(self.SPplacex[self.cline])+str(self.SPplacex[self.cline])+'\n'
                     +'G='+str(self.G[self.cline])+'\n')
        ax.set_xlim(-self.MAXboundary, self.MAXboundary)
        ax.set_ylim(-self.MAXboundary, self.MAXboundary)
        ax.set_zlim(0, 150)
        plt.self.show()
    def trajectory(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        trax = self.data[:, 0]
        tray = self.data[:, 1]
        ax.set_xlim(-self.MAXboundary , self.MAXboundary )
        ax.set_ylim(-self.MAXboundary , self.MAXboundary )
        ax.set_zlim(0, 120)
        ax.plot3D(trax, tray, 100*np.ones_like(trax), 'r')
        ax.scatter3D(self.SPplacex, self.SPplacey, np.zeros_like(self.SPplacex), 'g')
        for cline in range(self.NUAV):
            ax.text(self.SPplacex[cline], self.SPplacey[cline], 0,
                'loc=' + str(self.SPplacex[cline]) + str(self.SPplacey[cline]) + '\n'
                + 'G=' + str(self.G[cline]))
        plt.self.show()

    def render(self, mode='human'):
        return {}


