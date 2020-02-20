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
class DL2dwithblock(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 3
        self.Vmax = 20                                 # 最大速度    in m/s
        self.amax = 5                                # 最大加速度  in m^2/s
        self.MAXp = 400
        self.delta = 0.5                             # 时隙
        self.T = 1000                                  # 总时间
        self.time = 0

        self.N = self.T/self.delta
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
        self.blockstartx = [50, -100, -200]
        self.blockstarty = [150, -50, 0]
        self.blockw = [100, 80, 90]
        self.blockl = [70, 90, 60]

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0])                             # 加速度
        self.v = np.array([10, 10])                          # 速度
        self.place = np.array([0, 0])  # 加速度

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
        self.Grecord = np.copy(self.G0)
        self.P = 10                                              # 初始发射功率dBm
        self.P_data = 5                                          # 处理功率  单位W
        self.PLmax = self.PLoss()
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        # 定义动作空间
        self.a_sp = []
        achoose = np.linspace(-self.amax, self.amax, num=5)
        for i in range(0, len(achoose)):
            for j in range(0, len(achoose)):
                self.a_sp.append([achoose[i],achoose[j]])
        self.action_space = spaces.Discrete(len(achoose)**2)
        "利用了用户的信噪比 和 数据量 + 无人机位置 x,y 和 无人机速度 vx,vy "
        self.observation_space = spaces.Box(low=-150, high=150, shape=(4,), dtype=np.float32)
        # 定义输出的变量
        self.show = 0
        self.boundary = 1

    def reset(self):
        # 初始化强化学习
        self.done = 0
        self.time = 0
        # 初始化无人机 和 node节点 数据
        self.a = np.array([0, 0])  # 加速度
        self.v = np.array([10, 10])  # 速度

        inarea = 1
        while inarea:
            placex = np.random.randint(-self.MAXp, self.MAXp, 1)  # 节点位置x
            placey = np.random.randint(-self.MAXp, self.MAXp, 1)  # 节点位置y
            inarea = self.inblock(placex, placey)
        self.place = np.array([placex, placey])


        inarea = 1
        while inarea:
            goalx = np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置x
            goaly = np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置y
            inarea = self.inblock(goalx, goaly)
        self.goal = np.array([goalx, goaly])

        S = np.array([self.place[0]/self.MAXp, self.place[1]/self.MAXp])
        S = np.append(S, [v/self.Vmax for v in self.v])
        return S

    def step(self, a):
        reward = 0
        done = 0
        self.time +=1
        self.a = np.array(self.a_sp[a])
        self.P = self.Pmax
        P = 10**(self.P/10)/1000
        outbound = 0
        for i in range(len(self.v)):
            tempbound = self.bound(a[i] * self.delta + self.v[i], self.Vmax, -self.Vmax)
            self.v[i] += (1 - tempbound) * self.a[i] * self.delta
            outbound += tempbound
        reward += -(outbound) * 10

        x, y = [self.place[i] + self.v[i] * self.delta + 0.5 * self.a[i] * (self.delta ** 2) for i in
                   range(len(self.v))]
        tp1 = self.bound(x, self.MAXp, -self.MAXp)
        tp2 = self.bound(y, self.MAXp, -self.MAXp)
        tp4 = self.inblock(x, y)
        if tp4:
            reward += -(tp1 + tp2 + tp4) * 10
        else:
            olddis = sum([abs(self.goal[i] - self.place[i]) for i in range(len(self.goal))])
            self.place[0] = (1 - tp1) * x + tp1 * self.place[0],
            self.place[1] = (1 - tp2) * y + tp2 * self.place[1],
            # 判断所链接的用户
            dis = sum([abs(self.goal[i] - self.place[i]) for i in range(len(self.goal))])
            # 判断信噪比
            reward += olddis-dis

            '''
            the first reward
            '''
            # if output > 0:
            #     reward += output*10
            # else:
            #     reward += -teskleft*1e-2
        if dis < 10:
            done = 1
        S_ = np.array([self.place[0]/self.MAXp, self.place[1]/self.MAXp])
        S_ = np.append(S_, [v/self.Vmax for v in self.v])
        return S_, reward/100, done, {'G0': np.sum(self.G0)}
# 判断生成的点是否在可行范围内
    def inblock(self,x,y):
        blocknum = len(self.blockstartx)
        flag = 0
        for time in range(blocknum):
            if self.blockstartx[time] < x < self.blockstartx[time]+self.blockw[time] \
                    and self.blockstarty[time] < y < self.blockstarty[time]+self.blockl[time]:
                flag = 1
                break
        return flag
# 判断速度是否在可行范围内
    def bound(self, v, upband, lowband):
        flag = int(v > upband) + int(v < lowband)
        return flag
# 计算瞬时信噪比
    def Rate(self):
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[i]) ** 2 +
                          (self.placey - self.SPplacey[i]) ** 2 +
                          (self.placez) ** 2)
            K = (self.placey - self.SPplacey[i]) / (self.placex - self.SPplacex[i])
            flag = 0
            for bl in range(len(self.blockstartx)):
                dot1 = np.sign(K * (self.blockstartx[bl] - self.placex) - (self.blockstarty[bl] - self.placey))
                dot2 = np.sign(K * (self.blockstartx[bl] + self.blockw[bl] - self.placex) - (self.blockstarty[bl] - self.placey))
                dot3 = np.sign(K * (self.blockstartx[bl] - self.placex) - (self.blockstarty[bl] + self.blockl[bl] - self.placey))
                dot4 = np.sign(K * (self.blockstartx[bl] + self.blockw[bl] - self.placex) - (self.blockstarty[bl] + self.blockl[bl]- self.placey))
                if dot1 > 0 and dot2 > 0 and dot3 > 0 and dot4 > 0:
                    flag = 1
                elif dot1 < 0 and dot2 < 0 and dot3 < 0 and dot4 < 0:
                    flag = 1
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
        ax.set_xlim(-self.MAXp, self.MAXp)
        ax.set_ylim(-self.MAXp, self.MAXp)
        ax.set_zlim(0, 150)
        plt.self.show()
    def trajectory(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        trax = self.data[:, 0]
        tray = self.data[:, 1]
        ax.set_xlim(-self.MAXp , self.MAXp )
        ax.set_ylim(-self.MAXp , self.MAXp )
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


