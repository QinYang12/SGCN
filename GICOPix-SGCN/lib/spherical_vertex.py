import numpy as np
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
from math import isnan
import pdb

def cos(x):
    return np.cos(x*np.pi)
def sin(x):
    return np.sin(x*np.pi)

def normalize(v):
    d = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    for i in range(3):
        v[i]=v[i]/d
    return v

def subdivide(v1,v2,v3,tile_number,points,depth):
    depth = depth+1
    if tile_number == 0:
        return
    
    v12=[0,0,0]
    v23=[0,0,0]
    v13=[0,0,0]
    for i in range(3):
        v12[i] = (v1[i]+v2[i])/2
        v23[i] = (v2[i]+v3[i])/2
        v13[i] = (v1[i]+v3[i])/2
    v12 = normalize(v12)
    v23 = normalize(v23)
    v13 = normalize(v13)
    if v12 not in points:
        points.append(v12)
    if v23 not in points:
        points.append(v23)
    if v13 not in points:
        points.append(v13)
    subdivide(v1,v12,v13,tile_number-1,points,depth)
    subdivide(v2,v23,v12,tile_number-1,points,depth)
    subdivide(v3,v13,v23,tile_number-1,points,depth)
    subdivide(v12,v23,v13,tile_number-1,points,depth)
    #return depth

def uniform_sampling(m,r):
    #geodesic grid#
    #m is the number of vertex#
    #2**(2*tile_num)*10+2=m#
    #r is the radius#

    
    points_theta = []
    depth = 0
    tile_number = m
    '''initial define'''
    m = np.sqrt(50-10*np.sqrt(5))/10
    n = np.sqrt(50+10*np.sqrt(5))/10 
    v_base=[[-m,0,n],[m,0,n],[-m,0,-n],[m,0,-n],[0,n,m],[0,n,-m],[0,-n,m],[0,-n,-m],[n,m,0],[-n,m,0],[n,-m,0],[-n,-m,0]]
    points = v_base
    index = [[1,4,0],[4,9,0],[4,5,9],[8,5,4],[1,8,4],
             [1,10,8],[10,3,8],[8,3,5],[3,2,5],[3,7,2],
             [3,10,7],[10,6,7],[6,11,7],[6,0,11],[6,1,0],
             [10,1,6],[11,0,9],[2,11,9],[5,2,9],[11,2,7]]
    for i in range(20):
        subdivide(v_base[index[i][0]],v_base[index[i][1]],v_base[index[i][2]],tile_number,points,depth)
    print('the len of point'+str(len(points))+'the second'+str(points[0]))

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arccos(z/r)/np.pi
        phi = np.arctan(y/x)/np.pi
        if x < 0 :
            phi = phi+1
        if x >= 0 and y < 0 :
            phi = phi +2                    #theta:0~1, phi:0~2
        theta = round(0.5-theta,4)
        phi = round(phi-1,4)                 #theta:0.5~-0.5, phi:-1~1
        if x == 0 and y == 0 :
            phi = 0.0
        points_theta.append([theta,phi])
    return points, points_theta

def transform(v):

    v = np.array(v)
    x, y, z = v[:,0], v[:, 1], v[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    points_theta = np.transpose(np.array([lat/np.pi, long/np.pi]), [1,0])

    return points_theta   
 

def uniform_sampling_spiral(m,r): 

    #m is the number of vertex
    #spiral grid#
    points_xyz = []
    points_theta = []
    for i in range(m):
        h = -1+2.0*i/(m-1)
        theta=np.arccos(h)/np.pi
        if i==0 or i==m-1:
            phi=0
        else: 
            phi=(phi*np.pi+3.6/np.sqrt(m*(1-h**2)))%(2*np.pi)/np.pi  #the phi of the right side is the i-1 value
            phi=round(phi,4)
        theta = round(theta-0.5, 4)
        if phi>1:
            phi=phi-2
        x = r*sin(theta)*cos(phi)
        y = r*sin(theta)*sin(phi)
        z = r*cos(theta)
        points_xyz.append([x,y,z])
        points_theta.append([theta,phi])
    return points_xyz,points_theta


def project_point(imgs,radius,points,rotation=False,selfRotation=False, centers=None, points_raw=None):
    vertex = []
    if len(imgs.shape)>3:
        img_num, img_w, img_h, img_channel = imgs.shape
    elif len(imgs.shape)==3:
        img_num, img_w, img_h = imgs.shape
    elif len(imgs.shape)<3:
        print('imgs for dataset are unknown shape')
        pdb.set_trace()
    for i in range(imgs.shape[0]):
        beta = np.random.rand()*2        
        if rotation:
            theta_tangent = round(np.random.randint(-90,90)/180,4) #discrete integer in degree, -0.5~0.5
            phi_tangent = round(np.random.randint(-180,180)/180,4) #-1~1
        else:
            theta_tangent = round(np.random.randint(-45,45)/180,4)   #-0.5~0.5
            phi_tangent = round(np.random.randint(-90,90)/180,4)     #-1~1
        if centers is not None:
            theta_tangent, phi_tangent   = centers
        pixels = []
        theta_erp = []
        phi_erp = []
        for j in range(len(points)):
            theta = points[j][0]
            phi = points[j][1]
            distance = np.arccos(cos(theta)*cos(theta_tangent)*cos(phi-phi_tangent)+sin(theta)*sin(theta_tangent))
            c = sin(theta_tangent)*sin(theta)+cos(theta_tangent)*cos(theta)*cos(phi-phi_tangent)
            x = radius*cos(theta)*sin(phi-phi_tangent)/c
            y = radius*(cos(theta_tangent)*sin(theta)-sin(theta_tangent)*cos(theta)*cos(phi-phi_tangent))/c

            if selfRotation:  
                x1 = x*cos(beta)-y*sin(beta)
                y1 = y*cos(beta)+x*sin(beta) 
                x = x1
                y = y1
            if abs(x)>=img_w/2 or abs(y)>=img_h/2 or distance>0.5*np.pi or isnan(x):
                pixel = 0 * imgs[i,0,0]
            else:
                x = x+img_w/2
                y = img_h/2-y
                '''
                pixel = imgs[i, int(y),int(x)]
                '''
                # bilinear
                x_int = int(x)
                y_int = int(y)
                dx = x - x_int
                dy = y - y_int
                pixel_bo_left  = imgs[i, y_int, x_int]
                pixel_bo_right = imgs[i, y_int, min(x_int+1, img_w-1)]
                pixel_up_left  = imgs[i, min(y_int+1, img_h-1), x_int]
                pixel_up_right = imgs[i, min(y_int+1, img_h-1), min(x_int+1, img_w-1)]
                pixel_bo = (1-dx) * pixel_bo_left + dx * pixel_bo_right
                pixel_up = (1-dx) * pixel_up_left + dx * pixel_up_right
                pixel = (1-dy) * pixel_up + dy * pixel_bo
            pixels.append(pixel)
        if (i+1)%100==0:
            print('project_data for {}/{} done.'.format(i+1,img_num))
        vertex.append(pixels)
    return vertex





    

