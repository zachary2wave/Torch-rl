
#检查LoS径或者单反射径是否被障碍物（wall)遮挡

import numpy as np

def RTrackinglos(walls,startpoint,endpoint,wall_normal):
    vec=endpoint-startpoint
    flag=1
    for i in range(walls.shape[0]):
        for j in range(walls.shape[1]):
            dot1=np.dot(vec,wall_normal[i,j,:])
            if dot1==0:
                continue
            else:
                scale=np.dot(walls[i,j,:]-startpoint,wall_normal[i,j,:])/dot1
                if scale<1 and scale>0:
                    interset_point=vec*scale+startpoint
                    minmaxx=np.array([min(walls[i,j,0],walls[i,(j+1)%4,0]),max(walls[i,j,0],walls[i,(j+1)%4,0])])
                    minmaxy=np.array([min(walls[i,j,1],walls[i,(j+1)%4,1]),max(walls[i,j,1],walls[i,(j+1)%4,1])])
                    if minmaxx[0]==minmaxx[1] and np.prod(minmaxy-np.tile(interset_point[1],(1,2)))<0:#当墙壁平行于y轴时判断是否相交
                        flag=0
                        return flag
                    elif np.prod(minmaxx-np.tile(interset_point[0],(1,2)))<0 and minmaxy[0]==minmaxy[1]:#当墙壁平行于x轴时判断是否相交
                        flag=0
                        return flag
                    elif np.prod(minmaxx-np.tile(interset_point[0],(1,2)))<0 and np.prod(minmaxy-np.tile(interset_point[1],(1,2)))<0:
                        flag=0
                        return flag
    return flag


def RTracking_reflect(walls,startpoint,endpoint,wall_normal,index1,index2,Tx):
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
            if flag==1:
                flag1=RTrackinglos(walls,interset_point,endpoint,wall_normal)
                flag2=RTrackinglos(walls,Tx,interset_point,wall_normal)
                if flag1==0 or flag2==0:
                    flag=0

    return flag,interset_point













