from Angletest import beamFind
import numpy as np
import math
import cvxopt
import matplotlib.pyplot as plt
from multi_resolution_codebook import pcodebook

#配置参数
lens=36*2
index=np.array(range(lens))
Tx_xy=np.array([15,15])#设定发射端和接收端的位置信息
R=45
Rx=Tx_xy[0]+R*np.cos(index*360/lens*math.pi/180)+np.random.normal(0,2,lens)
Ry=Tx_xy[1]+R*np.sin(index*360/lens*math.pi/180)+np.random.normal(0,2,lens)
wall1=[[0,0],[10,0],[10,10],[0,10]]
wall2=[[5,20],[20,20],[20,30],[5,30]]
wall3=[[-5,18],[4,18],[4,25],[-5,25]]
wall4=[[15,-5],[50,-5],[50,5],[15,5]]
wall5=[[25,17],[45,17],[45,27],[25,27]]
walls=np.array([wall1,wall2,wall3,wall4,wall5])
num_wall=walls.shape[0]
wallx=[walls[i,:,0] for i in range(walls.shape[0])]
wally=[walls[i,:,1] for i in range(walls.shape[0])]

#设计多级码本
Nt=128
s=128
codebook=pcodebook(Nt,s)
print(codebook)

#绘制接收端运动路径
fig1=plt.figure()
for i in range(num_wall):
    plt.fill(wallx[i],wally[i],facecolor='b',alpha=0.5)
plt.plot(Tx_xy[0],Tx_xy[1],'s')
plt.plot(Rx,Ry,'r--s')
plt.show()

#针对不同的接收端位置求对应的信道模型，包括增益及角度信息
d_Rx=[Rx[(i+1)%lens]-Rx[i] for i in range(lens)]
d_Ry=[Ry[(i+1)%lens]-Ry[i] for i in range(lens)]
maxdx=max(d_Rx)
maxdy=max(d_Ry)
dT=10#相邻两个用户位置的时间间隔

snrdb=np.arange(-10.0,16.0,5.0)
snr=10.0**(snrdb/10.0)
trans_pow=snr[0]#此处设定噪声方差为1
level=1
ind=1

vxy=np.zeros([2,lens])#第一行存放vx，第二行存放vy
for i in range(lens):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%index :", i,'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    vxy[0,i]=d_Rx[i]/dT
    vxy[1,i]=d_Ry[i]/dT
    t=dT*i
    curr_chan=beamFind(Rx[i],Ry[i],Tx_xy,walls,num_wall,wallx,wally,vxy[0,i],vxy[1,i],t)
    #reward设定--------注释：ind是选择的码本序号,level表示选择的码本层级序号，ind为该码本层级中的第ind个码本
    opt_rece_sig_no_noise = math.sqrt(trans_pow) * cvxopt.blas.dot(codebook[level, ind], curr_chan)
    opt_rewd = np.log2(1 + np.abs(opt_rece_sig_no_noise) ** 2)

print('TEST END!')



