#!/usr/bin/python

import numpy as np
from get_cpp import *


def create_cpp(xi, yi, l, function, function_parameters, function_order, covariance_type, distance_conv):
    Cpp = np.empty((0, len(xi)))
    for i in xrange(0, len(function_parameters)):
        col1 = int(function_order[i,1])
        col2 = int(function_order[i,2])
        if (function_order[i,0].split('+')[0] != function_order[i,0].split('+')[1]) and (covariance_type != 'jl'):
            Cpp_p = np.zeros((len(xi), len(xi)))
        else:
            Cpp_p = get_cpp(xi, yi, function, function_parameters[i], distance_conv, covariance_type, function_order[i,0])
        Cpp = np.concatenate([Cpp, Cpp_p])
    Cpp1 = np.empty((0, len(xi) * l.shape[1]))
    for i in xrange(0, l.shape[1]):
        Cpp2 = np.empty((len(xi), 0))
        for j in xrange(0, l.shape[1]):
            Cpp2 = np.concatenate([Cpp2, Cpp[((i * l.shape[1] + j) * len(xi)):(((i * l.shape[1] + j) + 1) * len(xi)),:]], axis=1)
        Cpp1 = np.concatenate([Cpp1, Cpp2], axis=0)
    Cpp = Cpp1 + [0]
    return Cpp;

