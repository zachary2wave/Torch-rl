import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from gym.utils import seeding
# import panda as pd
import scipy.io as sio
import time as TT
'''
11.2 建立碰撞模型
11.3 修改碰撞

'''
class C2d_controlP_V0(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 3
        self.Placemax = 10
        self.delta = 0.5
        self.Pmax = 10                                 #  dBm  功率
        self.choose = 'Urban'
        self.SNRlimit = 0
        self.K = 0.01                                  # 空气阻力系数
        self.alpha = 4
        # 环境信道参数
        self.done = 0
        self.B = 1e6                                   # 带宽 1Mhz
        self.N0 = -130                                 # dBm
        self.m = 1900                                  # in g
        self.R_th = 1.3*self.B                         #

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)
        '''
        block 区域 其中a 为 左上点坐标 b为 右下点坐标 
        也就是ax一定小于bx  ay一定大于by
        '''
        self.blockax = [50]
        self.blockay = [200]
        self.blockbx = [100]
        self.blockby = [100]

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.placex = 0                                          # 无人机位置x
        self.placey = 0                                          # 无人机位置y
        self.placez = 100                                         # 无人机位置z
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
        self.a_sp=[]

        self.action_space = spaces.Box(low=-self.Placemax, high=self.Placemax, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-150, high=150, shape=(2*self.NSP+2,), dtype=np.float32)
        # 定义输出的变量
        self.show = 0
        self.boundary = 1

    def reset(self):
        # 初始化无人机 和 node节点 数据
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
        self.placez = 50  # 无人机位置z
        "重置节点位置"
        self.SPplacex = np.array([])
        self.SPplacey = np.array([])
        for i in range(self.NSP):
            inarea = 1
            while inarea:
                SPplacex = np.random.randint(-self.MAXboundary , self.MAXboundary , 1)  # 节点位置x
                SPplacey = np.random.randint(-self.MAXboundary , self.MAXboundary , 1)  # 节点位置y
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
        # repositionx, repositiony= self.relative_position()
        temp = np.concatenate(([g/300 for g in self.G],[snr/10 for snr in self.SNR]), axis=0)
        S = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S = np.append(S, [self.placex/self.MAXboundary , self.placey/self.MAXboundary ])

        self.totalR = self.rate[self.cline]*self.delta/1e6
        self.step_time = 0

        return S

    def step(self, a):
        reward = 0
        self.time += 1
        # outbound1 = self.inblock(self.placex + a[0], self.placey + a[1])
        outbound2 = self.placebound(self.placex + a[0], self.placey + a[1])
        if outbound2:
            reward += -(outbound2) * 10
        else:
            self.placex += a[0]   # 无人机位置x
            self.placey += a[1]   # 无人机位置y
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
            # if output > 0:
            #     reward = 1
            # else:
            #     reward = -1
            '''
            the first reward
            '''
            if output > 0:
                reward += output*self.delta*10
            else:
                reward += -1
        teskleft = np.sum(self.G)
        if teskleft == 0 or self.time == 10000:
            self.done = 1

        temp = np.concatenate(([g/300 for g in self.G], [snr/10 for snr in self.SNR]), axis=0)
        S_ = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S_ = np.append(S_, [self.placex/self.MAXboundary , self.placey/self.MAXboundary ])
        return S_, reward/100, self.done, {'G0':np.sum(self.G0)}
# 判断生成的点是否在可行范围内
    def inblock(self,x,y):
        blocknum = len(self.blockax)
        flag = 0
        for time in range(blocknum):
            if self.blockax[time] <x<self.blockbx[time] and self.blockby[time] < y < self.blockay[time]:
                flag = 1
                break
        return flag
# 判断速度是否在可行范围内
    def speedbound(self,vx,vy):
        flag = int(vx > self.Vmax)+int(vx < -self.Vmax)+int(vy > self.Vmax)+int(vy < -self.Vmax)
        return flag
    def placebound(self,x,y):
        flag = int(x > self.MAXboundary)+int(x < -self.MAXboundary)+int(y > self.MAXboundary)+int(y < -self.MAXboundary)
        return flag
# 计算瞬时信噪比
    def Rate(self):
        PLmax, D = self.PLoss()
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            SNR[i] = self.P-PLmax[i]-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR

# 计算瞬时时间延迟 loss
    def PLoss(self):
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

        PLmax = []
        for time in range(0, self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[time]) ** 2 +
                          (self.placey - self.SPplacey[time]) ** 2 +
                          (self.placez) ** 2)
            K = (self.placey - self.SPplacey[time])/(self.placex - self.SPplacex[time])
            flag = 0
            for bl in range(len(self.blockax)):
                dot1 = np.sign(K * (self.blockax[bl] - self.placex) - (self.blockay[bl] - self.placey))
                dot2 = np.sign(K * (self.blockax[bl] - self.placex) - (self.blockby[bl] - self.placey))
                dot3 = np.sign(K * (self.blockbx[bl] - self.placex) - (self.blockby[bl] - self.placey))
                dot4 = np.sign(K * (self.blockbx[bl] - self.placex) - (self.blockay[bl] - self.placey))
                if dot1>0 and dot2>0 and dot3>0 and dot4>0:
                    flag = 1
                elif dot1<0 and dot2<0 and dot3<0 and dot4<0:
                    flag = 1
            PLmax.append(10 * math.log10(D**self.alpha) + self.lossb + (1-flag)*inta_los+ flag * inta_Nlos)
        return PLmax, D


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
        ax = fig.gac()
        trax = self.data[:, 0]
        tray = self.data[:, 1]

        ax.plot3D(trax, tray, 100*np.ones_like(trax), 'r')
        ax.scatter3D(self.SPplacex, self.SPplacey, np.zeros_like(self.SPplacex), 'g')
        for cline in range(self.NUAV):
            ax.text(self.SPplacex[cline], self.SPplacey[cline], 0,
                'loc=' + str(self.SPplacex[cline]) + str(self.SPplacey[cline]) + '\n'
                + 'G=' + str(self.G[cline]))
        plt.self.show()

    def render(self, mode='human'):
        return {}
