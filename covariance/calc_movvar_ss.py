#!/usr/bin/python

import numpy as np
from pyproj import Geod
g = Geod(ellps='WGS84')

def calc_movvar_ss(x, y, obs, function_order, d_conv, delta_mov, min_num=2, fill_val=9999, C0=np.array([0])):
    '''
    Calculate moving variances depending on delta
    '''
    
    c1_comp = np.empty((len(x), 0)); c2_comp = np.empty((len(x), 0))
    for k in xrange(0, len(function_order)):
        col1 = int(function_order[k,1])
        col2 = int(function_order[k,2])
        if C0.all() == 0:
            c01 = np.sum(obs[:,col1]**2) / len(obs[:,col1])
            c02 = np.sum(obs[:,col2]**2) / len(obs[:,col2])
            C0_k = c01 * 0.5 + c02 * 0.5
        else:
            C0_k = C0[k]
        if k == 0:
            c1 = []
            c2 = []
        if k == 1:
            c1 = c1_comp[:,0]
            c2 = []
        if k == 2:
            c1 = c2_comp[:,1]
            c2 = c1_comp[:,0]
        if k == 3:
            c1 = c2_comp[:,1]
            c2 = c2_comp[:,1]
        if len(c1) == 0:
            c1 = []
            for i in xrange(0, len(x)):
                dij = []
                for j in xrange(0, len(x)):
                    if d_conv == 'sphere':
                        dij.append(g.inv(x[i], y[i], x[j], y[j])[2] / 1000.)
                    elif d_conv == 'utm':
                        dij.append(np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2))
                dij = np.asarray(dij)
                dij_mask = np.ma.masked_less_equal(np.abs(dij), delta_mov).mask
                if np.sum(dij_mask) == 0:
                    data1_delta = np.array([np.sqrt(C0_k / 2.), np.sqrt(C0_k / 2.)])
                elif 0 < np.sum(dij_mask) < min_num:
                    if fill_val == 9999:
                        fill_val1 = (np.mean(obs[dij_mask, col1]) + C0_k) / 2.
                    else:
                        fill_val1 = fill_val
                    data1_delta = np.vstack((np.reshape(obs[dij_mask, col1], (obs[dij_mask, col1].shape[0], 1)), np.zeros(((min_num  - obs[dij_mask, col1].shape[0]), 1)) + fill_val1))
                else:
                    data1_delta = obs[dij_mask, col1]
                c1.append(np.sqrt(np.sum(data1_delta**2) / float(len(data1_delta) - 1)))
            c1 = np.asarray(c1)
        if len(c2) == 0:
            c2 = []
            for i in xrange(0, len(x)):
                dij = []
                for j in xrange(0, len(x)):
                    if d_conv == 'sphere':
                        dij.append(g.inv(x[i], y[i], x[j], y[j])[2] / 1000.)
                    elif d_conv == 'utm':
                        dij.append(np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2))
                dij = np.asarray(dij)
                dij_mask = np.ma.masked_less_equal(np.abs(dij), delta_mov).mask
                if np.sum(dij_mask) == 0:
                    data2_delta = np.array([np.sqrt(C0_k / 2.), np.sqrt(C0_k / 2.)])
                elif 0 < np.sum(dij_mask) < min_num:
                    if fill_val == 9999:
                        fill_val2 = (np.mean(obs[dij_mask, col2]) + C0_k) / 2.
                    else:
                        fill_val2 = fill_val
                    data2_delta = np.vstack((np.reshape(obs[dij_mask, col2], (obs[dij_mask, col2].shape[0], 1)), np.zeros(((min_num  - obs[dij_mask, col2].shape[0]), 1)) + fill_val2))
                else:
                    data2_delta = obs[dij_mask, col2]
                c2.append(np.sqrt(np.sum(data2_delta**2) / float(len(data2_delta) - 1)))
            c2 = np.asarray(c2)
        c1_comp = np.c_[c1_comp, c1]
        c2_comp = np.c_[c2_comp, c2]
    return c1_comp, c2_comp;
