#用于测试之前代码正确与否的测试函数，即仅考虑单反射交点在平面上，但是不考虑入射射线和反射射线被平面阻塞情况
import numpy as np

def RTracking_reflectVerif(walls,startpoint,endpoint,wall_normal,index1,index2):
    vec=endpoint-startpoint
    flag=0
    interset_point=np.array([0,0])
    dot1 = np.dot(vec, wall_normal[index1, index2, :])
    if dot1==0:
        return flag,interset_point
    else:
        scale = np.dot(walls[index1, index2, :] - startpoint, wall_normal[index1, index2, :]) / dot1
        interset_point = vec * scale + startpoint
        if scale < 1 and scale > 0:
            minmaxx = np.array([min(walls[index1, index2, 0], walls[index1, (index2 + 1) % 4, 0]), max(walls[index1, index2, 0], walls[index1, (index2+ 1) % 4, 0])])
            minmaxy = np.array([min(walls[index1, index2, 1], walls[index1, (index2 + 1) % 4, 1]), max(walls[index1, index2, 1], walls[index1, (index2 + 1) % 4, 1])])
            if minmaxx[0] == minmaxx[1] and np.prod(minmaxy - np.tile(interset_point[1], (1, 2))) < 0:  # 当墙壁平行于y轴时判断是否相交
                flag=1
            elif np.prod(minmaxx - np.tile(interset_point[0], (1, 2))) < 0 and minmaxy[0] == minmaxy[1]:  # 当墙壁平行于x轴时判断是否相交
                flag=1
            elif np.prod(minmaxx - np.tile(interset_point[0], (1, 2))) < 0 and np.prod(minmaxy - np.tile(interset_point[1], (1, 2))) < 0:
                flag=1
    return flag,interset_point