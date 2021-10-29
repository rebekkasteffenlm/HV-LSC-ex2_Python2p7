#!/usr/bin/python

import numpy as np
from get_cps import *


def create_cps(x, y, xi, yi, l, function, function_parameters, function_order, covariance_type, distance_conv):
    Cps = np.empty((0, len(x)))
    for i in xrange(0, len(function_parameters)):
        col1 = int(function_order[i,1])
        col2 = int(function_order[i,2])
        if (function_order[i,0].split('+')[0] != function_order[i,0].split('+')[1]) and (covariance_type != 'jl'):
            Cps_p = np.zeros((len(xi), len(x)))
        else:
            Cps_p = get_cps(x, y, xi, yi, function, function_parameters[i], distance_conv, covariance_type, function_order[i,0])
        Cps = np.concatenate([Cps, Cps_p])
    Cps1 = np.empty((0, len(x) * l.shape[1]))
    for i in xrange(0, l.shape[1]):
        Cps2 = np.empty((len(xi), 0))
        for j in xrange(0, l.shape[1]):
            Cps2 = np.concatenate([Cps2, Cps[((i * l.shape[1] + j) * len(xi)):(((i * l.shape[1] + j) + 1) * len(xi)),:]], axis=1)
        Cps1 = np.concatenate([Cps1, Cps2], axis=0)
    Cps = Cps1 + [0]
    return Cps;

