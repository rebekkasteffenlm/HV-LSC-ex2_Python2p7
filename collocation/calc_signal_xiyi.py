#!/usr/bin/python

import numpy as np
from calc_sp import *
from calc_dsp import *
from covariance import *

def calc_signal_xiyi(x, y, xi, yi, l, Css, Czz_m1, function, function_parameters, function_order, covariance_type, distance_conv, new_point_limit, k, mov_var=[0], function_parameter=[1.]):
    '''
    Calculate signal at unknown points
    '''
    num_loop = int(np.ceil(len(xi) / float(new_point_limit)))
    l_new = np.empty((0, 1))
    for i in xrange(0, l.shape[1]):
        l_new = np.concatenate((l_new, np.reshape(l[:,i], (len(l), 1))), axis=0)
    min_range = k * new_point_limit
    if k == num_loop - 1:
        max_range = len(xi)
    else:
        max_range = (k + 1) * new_point_limit
    
    Cps = create_cps(x, y, xi[min_range:max_range], yi[min_range:max_range], l, function, function_parameters, function_order, covariance_type, distance_conv)
    Cpp = create_cpp(xi[min_range:max_range], yi[min_range:max_range], l, function, function_parameters, function_order, covariance_type, distance_conv)
    if len(mov_var) > 1:
        if abs(l.min()) > abs(l.max()):
            max_l = abs(l.min())
        else:
            max_l = abs(l.max())
        if function_parameter[:,0].max() <= 1:
            max_l = 1.
        c1_ps = calc_movvar_ps(x, y, xi[min_range:max_range], yi[min_range:max_range], l / max_l, function_order, function_parameters, distance_conv, mov_var[0], mov_var[1], mov_var[2], mov_var[3])
        c2_ps = mov_var[4]
        c1_pp = c1_ps + [0]
        c2_pp = c1_pp[:,0].reshape((len(c1_pp), 1)) + [0]
        for i in xrange(c1_pp.shape[1] / l.shape[1] - 1):
            c2_pp = np.c_[c2_pp, c1_pp[:,l.shape[1]]]
        for i in xrange(l.shape[1] - 1):
            c2_pp = np.c_[c2_pp, c2_pp]
        
        movvar = np.empty((0, len(x)))
        for k in xrange(0, len(function_order)):
            movvar = np.concatenate([movvar, np.outer(c1_ps[:,k], c2_ps[:,k])])
        movvar1 = np.empty((0, len(x) * l.shape[1]))
        for i in xrange(0, l.shape[1]):
            movvar2 = np.empty((len(xi[min_range:max_range]), 0))
            for j in xrange(0, l.shape[1]):
                movvar2 = np.concatenate([movvar2, movvar[((i * l.shape[1] + j) * len(xi[min_range:max_range])):(((i * l.shape[1] + j) + 1) * len(xi[min_range:max_range])),:]], axis=1)
            movvar1 = np.concatenate([movvar1, movvar2], axis=0)
        movvar_cps = movvar1 + [0]
        Cps_mov = Cps * movvar_cps
        
        movvar = np.empty((0, len(xi[min_range:max_range])))
        for k in xrange(0, len(function_order)):
            movvar = np.concatenate([movvar, np.outer(c1_pp[:,k], c2_pp[:,k])])
        movvar1 = np.empty((0, len(xi[min_range:max_range]) * l.shape[1]))
        for i in xrange(0, l.shape[1]):
            movvar2 = np.empty((len(xi[min_range:max_range]), 0))
            for j in xrange(0, l.shape[1]):
                movvar2 = np.concatenate([movvar2, movvar[((i * l.shape[1] + j) * len(xi[min_range:max_range])):(((i * l.shape[1] + j) + 1) * len(xi[min_range:max_range])),:]], axis=1)
            movvar1 = np.concatenate([movvar1, movvar2], axis=0)
        movvar_cpp = movvar1 + [0]
        Cpp_mov = Cpp * movvar_cpp
        
        Cps = Cps_mov + [0]
        Cpp = Cpp_mov + [0]
        del Cps_mov; del Cpp_mov
    
    signal_xiyi_part = calc_sp(Cps, Czz_m1, l_new)
    signal_xiyi = np.empty((len(xi[min_range:max_range]), 0))
    for i in xrange(0, l.shape[1]):
        signal_xiyi = np.concatenate((signal_xiyi, signal_xiyi_part[(i * len(xi[min_range:max_range])):((i + 1) * len(xi[min_range:max_range]))]), axis=1)
    signal_xiyi_error_part = np.reshape(calc_dsp(Cps, Cpp, Czz_m1), signal_xiyi_part.shape)
    if len(mov_var) > 1:
        signal_xiyi_error_part = signal_xiyi_error_part * np.sqrt(np.mean(function_parameter[:,0]))
    signal_xiyi_error = np.empty((len(xi[min_range:max_range]), 0))
    for i in xrange(0, l.shape[1]):
        signal_xiyi_error = np.concatenate((signal_xiyi_error, signal_xiyi_error_part[(i * len(xi[min_range:max_range])):((i + 1) * len(xi[min_range:max_range]))]), axis=1)
    
    return signal_xiyi, signal_xiyi_error;
