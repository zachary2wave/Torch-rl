import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
# import panda as pd
import scipy.io as sio
import time as TT
'''
11.2 建立碰撞模型

'''
class C2d(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 7
        self.Vmax = 20                                 # 最大速度    in m/s
        self.Vmaxz = 10
        self.amax = 5                                # 最大加速度  in m^2/s
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
        self.blockstartx = [50, -200, -200, 100, -250]
        self.blockstarty = [150, -200, 0, -150, 200]
        self.blockw = [100, 160, 70,80, 130]
        self.blockl = [140, 80, 170,130, 40]

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0],dtype=np.float64)                             # 加速度
        self.v = np.array([10, 10],dtype=np.float64)                          # 速度
        self.placex = 0                                          # 无人机位置x
        self.placey = 0                                          # 无人机位置y
        self.placez = 50
        self.MAXboundary = 400

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
        self.G0 = np.random.uniform(10, 30, self.NSP)            # 每个节点的数据量 M为单位
        self.Grecord = np.copy(self.G0)
        self.P = 10                                              # 初始发射功率dBm
        self.P_data = 5                                          # 处理功率  单位W
        # self.PLmax = self.PLoss()
        self.rate, self.SNR, self.ref = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        # 定义动作空间
        self.action_space = spaces.Box(low=np.array([-self.amax, -self.amax]), \
                                       high=np.array([self.amax, self.amax]), \
                                       dtype=np.float32)
        "利用了用户的信噪比 和 数据量 + 无人机位置 x,y 和 无人机速度 vx,vy "
        self.observation_space = spaces.Box(low=-150, high=150, shape=(2*self.NSP+4,), dtype=np.float32)
        # 定义输出的变量
        self.show_flag = 0
        self.boundary = 1

        "for show the tra"
        self.tra = [[],[]]
        self.are = [[],[]]
        self.connect = []


    def reset(self):
        # 初始化强化学习
        self.done = 0
        # 初始化无人机 和 node节点 数据
        self.a = np.array([0, 0],dtype=np.float64)  # 加速度
        self.v = np.array([10, 10],dtype=np.float64) # 速度
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
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
        self.G0= np.random.uniform(10, 30, self.NSP)
        self.G = np.copy(self.G0)
        self.time = 0
        self.rate, self.SNR, self.ref = self.Rate()
        self.cline = np.argmax(self.SNR)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1

        # 强化学习部分重置
        self.time = 0
        self.totalR = self.rate[self.cline] * self.delta / 1e6

        temp = np.concatenate(([g/300 for g in self.G],[snr/10 for snr in self.SNR]), axis=0)
        S = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S = np.append(S, [self.placex/self.MAXboundary, self.placey/self.MAXboundary])
        S = np.append(S, [v/self.Vmax for v in self.v])

        self.tra[0].append(self.placex)
        self.tra[1].append(self.placey)
        self.are[0].append(0)
        self.are[1].append(0)
        self.connect.append(0)
        return S

    def step(self, a):
        ##############"initial part"################
        reward = 0
        self.cline = -1
        self.done = 0
        ########################################################
        self.time +=1
        self.a = a
        self.P = self.Pmax
        P = 10**(self.P/10)/1000
        outbound = 0
        v0 = np.copy(self.v)
        for i in range(len(self.v)):
            tempbound = self.bound(a[i] * self.delta + self.v[i], self.Vmax, -self.Vmax)
            self.v[i] = self.v[i] + float(1 - tempbound) * self.a[i] * self.delta
            outbound += tempbound
        reward += -(outbound) * 10
        # print(v0,"\t",self.v,"\t outbound:",outbound)
        x = self.placex + v0[0] * self.delta + 0.5 * self.a[0] * (self.delta**2)
        y = self.placey + v0[1] * self.delta + 0.5 * self.a[1] * (self.delta**2)
        tp1 = self.bound(x, self.MAXboundary, -self.MAXboundary)
        tp2 = self.bound(y, self.MAXboundary, -self.MAXboundary)
        tp4 = self.inblock(x, y)
        tp5 = self.inline(x, y)
        # print([self.placex, self.placey], "\t", [x, y],
              # "\t inblock:",tp4, "\t bound:",[tp2,tp4])

        if tp4:
            reward += -(tp1 + tp2 + tp4) * 10
            if tp5==1 or tp5==2:
                self.v[0]=0
            elif tp5==3 or tp5 == 4 :
                self.v[1]==0
        else:
            self.placex = (1 - tp1) * x + tp1 * self.placex  # 无人机位置x
            self.placey = (1 - tp2) * y + tp2 * self.placey  # 无人机位置y
            if tp1 == 1:
                self.v[0] = 0
            if tp2 == 1:
                self.v[1] = 0
            reward += -(tp1 + tp2) * 10
            # 判断所链接的用户
            self.rate, self.SNR, self.ref = self.Rate()
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
        teskleft = np.sum(self.G)
        if teskleft == 0 or self.time == 10000:
            self.done = 1

        temp = np.concatenate(([g/300 for g in self.G], [snr/10 for snr in self.SNR]), axis=0)
        S_ = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S_ = np.append(S_, [self.placex/self.MAXboundary , self.placey/self.MAXboundary ])
        S_ = np.append(S_, [v/self.Vmax for v in self.v])

        self.tra[0].append(self.placex)
        self.tra[1].append(self.placey)
        self.are[0].append(a[0])
        self.are[1].append(a[1])
        self.connect.append(self.cline)

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
# 判断速度是否在可行范围内

    def bound(self, v, upband, lowband):
        flag = int(v > upband) + int(v < lowband)
        return flag

    def inline(self, aimplacex, aimplacey):

        uavin = self.inblock(self.placex,self.placey)
        aimin = self.inblock(aimplacex,aimplacey)
        K = ((self.placey - aimplacey) + 1e-4) / ((self.placex - aimplacex) + 1e-4)
        B = (self.placey - K * self.placex)
        "第一个if 如果环境正确的话UAV是不会在block中的，但是此处保留了 以预防这个情况发生"
        "第二个if 这里是为了化简判断规则，就是当目标点已经在block中的时候，可以直接找到这个block，然后找相交的边"
        "第三个if 是为了UAV正规规划的，先找到离最近的block 然后再判断"
        flag = 0
        if uavin and aimin:
            flag = 10
            blocknum = uavin
        elif uavin or aimin:
            xmin, xmax = self.blockstartx[aimin-1], self.blockstartx[aimin-1] + self.blockw[aimin-1]
            ymin, ymax = self.blockstarty[aimin-1], self.blockstarty[aimin-1]+self.blockl[aimin-1]
            yinline = K * self.blockstartx[aimin-1]+B
            if ymin< yinline< ymax and min([self.placey,aimplacey]) < yinline < max([self.placey,aimplacey]):
                flag = 1
            yinline = K * (self.blockstartx[aimin-1]+self.blockw[aimin-1])+B
            if ymin < yinline < ymax and \
               min([self.placey, aimplacey]) < yinline < max([self.placey, aimplacey]):
                flag = 2
            xinline = (self.blockstarty[aimin-1]-B)/K
            if xmin < xinline < xmax and \
                min([self.placex, aimplacex]) < xinline < max([self.placex, aimplacex]):
                flag = 3
            xinline = (self.blockstarty[aimin-1]+self.blockl[aimin-1]-B)/K
            if xmin < xinline < xmax and \
                min([self.placex, aimplacex]) < xinline < max([self.placex, aimplacex]):
                flag = 4
        else:
            for bl in range(len(self.blockstartx)):
                xmin, xmax = self.blockstartx[bl], self.blockstartx[bl]+self.blockw[bl]
                ymin, ymax = self.blockstarty[bl], self.blockstarty[bl]+self.blockl[bl]
                yinline = K * self.blockstartx[bl]+B
                if ymin< yinline< ymax and min([self.placey,aimplacey]) < yinline < max([self.placey,aimplacey]):
                    flag = 1
                yinline = K * (self.blockstartx[bl]+self.blockw[bl])+B
                if ymin < yinline < ymax and \
                   min([self.placey, aimplacey]) < yinline < max([self.placey, aimplacey]):
                    flag = 2
                xinline = (self.blockstarty[bl]-B)/K
                if xmin < xinline < xmax and \
                    min([self.placex, aimplacex]) < xinline < max([self.placex, aimplacex]):
                    flag = 3
                xinline = (self.blockstarty[bl]+self.blockl[bl]-B)/K
                if xmin < xinline < xmax and \
                    min([self.placex, aimplacex]) < xinline < max([self.placex, aimplacex]):
                    flag = 4

        return flag

# 计算瞬时信噪比
    def Rate(self):
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        ref = []
        numblock=len(self.blockstartx)
        for i in range(self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[i]) ** 2 +
                          (self.placey - self.SPplacey[i]) ** 2 +
                          (self.placez) ** 2)
            flag = int(bool(self.inline(self.SPplacex[i], self.SPplacey[i])))
            PLmax = 10 * math.log10(D ** self.alpha) + self.lossb + (1-flag) * self.inta_los\
                    + flag * self.inta_Nlos
            SNR[i] = self.P-PLmax-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR, ref


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

    def render(self, mode='human'):
        return {}

    def show_(self, save_gif=None):

        def plot_linear_squard(x, y, dx, dy, color='red'):
            xx = [x, x, x + dx, x + dx, x]
            yy = [y, y + dy, y + dy, y, y]
            kwargs = {'alpha': 1, 'color': color}
            block = ax.plot(xx, yy, **kwargs)
            return block
        fig = plt.figure()
        # plt.ion()
        ax = fig.gca()
        for i in range(len(self.blockstartx)):
            plot_linear_squard(self.blockstartx[i], self.blockstarty[i],
                               self.blockw[i], self.blockl[i], color='red')
        plt.scatter(self.SPplacex, self.SPplacey)
        for ki in range(self.NSP):
            plt.text(self.SPplacex[ki], self.SPplacey[ki], str(ki) )
            # + '-G=' + str(round(self.G[ki], 1))
        plt.xlim(-400, 400)
        plt.ylim(-400, 400)

        trajectory, = plt.plot([], [], c='r')
        arrow = plt.arrow([], [], [], [])
        line, = plt.plot([], [], '--')
        arrow = ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
        def init():
            return trajectory

        tra = self.tra
        are = self.are
        connect = self.connect
        SPplacex = self.SPplacex
        SPplacey = self.SPplacey
        def update(i):
            # print(i)
            # print(tra[0][0:i], tra[1][0:i])
            trajectory.set_data(tra[0][0:i], tra[1][0:i])
            if connect[i] > -1:
                line.set_data([tra[0][i], SPplacex[connect[i]]], [tra[1][i], SPplacey[connect[i]]])
            if i%5==0:
                arrow = ax.annotate("", xy=(tra[0][i]+are[0][i] * 10, tra[1][i]+are[1][i] * 10),
                                    xytext=(tra[0][i], tra[1][i]), arrowprops=dict(arrowstyle="->"))
            return trajectory, line
        ani = animation.FuncAnimation(fig, update, frames = range(len(self.tra[0])), init_func=init, interval=20)
        plt.show()
        if save_gif is not None:
            ani.save(save_gif+'.mp4', fps=36)


