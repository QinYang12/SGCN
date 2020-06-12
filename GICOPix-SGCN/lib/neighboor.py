import numpy as np
import argparse
import os
import copy
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 
import time
import pdb

def cos(x):
    return np.cos(x*np.pi)

def sin(x):
    return np.sin(x*np.pi)

def _nearest_9(points, center, left, right, idx_candidates):
    # search neghbor in bottom 3 pixels
    res  = []
    for j in idx_candidates:
        if len(res) >= 3:
            break
        theta = points[j][0]
        if theta > left and theta < right:
            res.append(j)
            continue
        theta = points[j][0] + 2
        if theta > left and theta < right:
            res.append(j)
            continue
        
    # if multi neighbors
    if len(res) < 1:
        return None
        pdb.set_trace()
    elif len(res) == 2:
        if abs(points[res[0]][0] - center) < abs(points[res[1]][0] - center):
            res = res[:1]
        else:
            res = res[1:]
    elif len(res) == 3:
        res = res[1:2]
    res = [res[0]-1, res[0], res[0]+1]
    return res


def _nearest_7(points, center, left, right, idx_candidates):
    # search neghbor in bottom 3 pixels
    res  = []
    for j in idx_candidates:
        if len(res) >= 3:
            break
        theta = points[j][0]
        if theta > left and theta < right:
            res.append(j)
            continue
        theta = points[j][0] + 2
        if theta > left and theta < right:
            res.append(j)
            continue
        
    # if multi neighbors
    if len(res) < 1:
        return None
        pdb.set_trace()
    elif len(res) == 1:
        if points[res[0]][0] - center > 0:
            res.append(res[0]+1)
        else:
            res = [res[0]-1, res[0]]
    elif len(res) == 3:
        ps = []
        ns = []
        for j in res:
            if points[j][0] - center > 0:
                ps.append(j)
            else:
                ns.append(j)
        if len(ps) ==3:
            b = ps[0]
            for a in ps[1:]:
                if abs(points[a][0]-center) < abs(points[b][0]-center):
                    b = a
            res = [b-1, b]
        elif len(ps)==2:
            a,b = ps
            if abs(points[a][0]-center) < abs(points[b][0]-center):
                res = [ns[0], a]
            else:
                res = [ns[0], b]
        elif len(ps)==1:
            a,b = ns
            if abs(points[a][0]-center) < abs(points[b][0]-center):
                res = [a, ps[0]]
            else:
                res = [b, ps[0]]
        elif len(ps)==0:
            b = ns[0]
            for a in ns[1:]:
                if abs(points[a][0]-center) < abs(points[b][0]-center):
                    b = a
            res = [b, b+1]


    return res

def neighbors_init(kernel_size==9,points_num=4096, points):
    #num = kernel_size ** 2
    num = 9

    # first 9 points
    if kernel_size == 7:
        idxs = [
        [ 1, 6, 2, 0, 5, 3, 4],
        [ 6, 0, 7, 1, 2, 8, 9],
        [ 1, 0, 9, 2, 3,10,11],
        [ 0, 5, 2, 3, 4,12,13],
        [ 0, 6, 3, 4, 5,14,15],
        [ 0, 1, 4, 5, 6,15,16],
        [ 0, 1, 5, 6, 7,17,18],
        [ 0, 1, 6, 7, 8,18,19],
        [ 1, 2, 7, 8, 9,20,21]
        ]
    elif kernel_size == 9:
        idxs = [
        [ 2, 3, 4, 1, 0, 5, 8, 7, 6],
        [ 6, 0, 3, 7, 1, 2, 8, 9,10],
        [ 8, 1, 0, 9, 2, 3,10,11,12],
        [ 1, 0, 5, 2, 3, 4,11,12,13],
        [ 2, 0, 6, 3, 4, 5,13,14,15],
        [ 3, 0, 7, 4, 5, 6,14,15,16],
        [ 4, 0, 1, 5, 6, 7,16,17,18],
        [ 5, 0, 1, 6, 7, 8,18,19,20],
        [ 0, 1, 2, 7, 8, 9,19,20,21]
        ]
                           
    for i,(theta,phi) in enumerate(points):
        if i < num or i >= (points_num - num):
            continue
        nes = []

        left = points[i-1][0]
        right= points[i+1][0]
        if left>right:
            right += 2

        if kernel_size==7:
            _nearest = _nearest_7
        elif kernel_size==9:
            _nearest = _nearest_9

        # search neghbor in top 3 pixels
        idx_candidates = i - 2 - np.array(range(min(20, i-2)))
        up_center_idx = _nearest(points, theta, left, right, idx_candidates)
        while up_center_idx is None:
            left  -= 0.1
            right += 0.1
            up_center_idx = _nearest(points, theta, left, right, idx_candidates)

        # search neghbor in bottom 3 pixels
        idx_candidates = i + 2 + np.array(range(min(20, points_num - 2 - i)))
        bo_center_idx = _nearest(points, theta, left, right, idx_candidates)
        while bo_center_idx is None:
            left  -= 0.1
            right += 0.1
            bo_center_idx = _nearest(points, theta, left, right, idx_candidates)

        # kernel input idx
        nes = up_center_idx + [i-1, i, i+1] + bo_center_idx
        idxs.append(nes)
    
    # last 9 points
    for i in range(9):
        tmp_idxs = (points_num - 1 - np.array(copy.deepcopy(idxs[8-i]))).tolist()
        idxs.append(tmp_idxs)


    # turn to (theta,phi)
    idxs = np.array(idxs).reshape(-1).tolist()
    #neighbors = map(lambda x: points[x], idxs)
    #neighbors = np.array(neighbors).reshape(points_num, num, 2)
    neighbors = np.array(idxs).astype(np.int32)
    return neighbors