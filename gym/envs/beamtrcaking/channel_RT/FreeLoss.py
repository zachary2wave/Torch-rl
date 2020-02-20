#计算自由空间中的路径损耗大小



import math

def Freeloss(d,f):#f单位是Hz，d单位是m
    n=2
    FSPL=20*math.log10(4*math.pi/3e8)+20*math.log10(f)
    PL=FSPL+10*n*math.log10(d)#单位dB
    PL=10**(PL/10)
    return PL