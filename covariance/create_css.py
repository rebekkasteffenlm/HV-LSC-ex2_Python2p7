#!/usr/bin/python

import numpy as np
from get_css import *


def create_css(x, y, l, function, function_parameters, function_order, covariance_type, distance_conv):
    Css = np.empty((0, len(x)))
    for i in xrange(0, len(function_parameters)):
        col1 = int(function_order[i,1])
        col2 = int(function_order[i,2])
        if (function_order[i,0].split('+')[0] != function_order[i,0].split('+')[1]) and (covariance_type != 'jl'):
            Css_p = np.zeros((len(x), len(x)))
        else:
            Css_p = get_css(x, y, function, function_parameters[i], distance_conv, covariance_type, function_order[i,0])
        Css = np.concatenate([Css, Css_p])
    Css1 = np.empty((0, len(x) * l.shape[1]))
    for i in xrange(0, l.shape[1]):
        Css2 = np.empty((len(x), 0))
        for j in xrange(0, l.shape[1]):
            Css2 = np.concatenate([Css2, Css[((i * l.shape[1] + j) * len(x)):(((i * l.shape[1] + j) + 1) * len(x)),:]], axis=1)
        Css1 = np.concatenate([Css1, Css2], axis=0)
    Css = Css1 + [0]
    return Css;

