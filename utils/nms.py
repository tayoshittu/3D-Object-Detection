# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from pc_util import bbox_corner_dist_measure

# boxes are axis aigned 2D boxes of shape (n,5) in FLOAT numbers with (x1,y1,x2,y2,score)
''' Ref: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
Ref: https://github.com/vickyboy47/nms-python/blob/master/nms.py 
'''
def nms_2d(boxes, overlap_threshold):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)
        suppress = [last-1]
        for pos in range(last-1):
            j = I[pos]
            xx1 = max(x1[i],x1[j])
            yy1 = max(y1[i],y1[j])
            xx2 = min(x2[i],x2[j])
            yy2 = min(y2[i],y2[j])
            w = xx2-xx1
            h = yy2-yy1
            if (w>0 and h>0):
                o = w*h/area[j]
                print('Overlap is', o)
                if (o>overlap_threshold):
                    suppress.append(pos)
        I = np.delete(I,suppress)
    return pick

def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        if old_type:
            o = (w*h)/area[I[:last-1]]
        else:
            inter = w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

def nms_crnr_dist(boxes, conf, overlap_threshold):
        
    I = np.argsort(conf)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)        
        
        scores = []
        for ind in I[:-1]:
            scores.append(bbox_corner_dist_measure(boxes[i,:], boxes[ind, :]))

        I = np.delete(I, np.concatenate(([last-1], np.where(np.array(scores)>overlap_threshold)[0])))

    return pick

