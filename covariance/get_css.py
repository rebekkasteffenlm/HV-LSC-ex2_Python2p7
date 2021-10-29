#!/usr/bin/python

import sys
import itertools
import numpy as np
from calc_fee import *
from calc_fen import *
from calc_fne import *
from calc_fnn import *
from function_covariance import *
from rest import *
from pyproj import Geod
R = 6371000
g = Geod(ellps='WGS84')

def get_css(x, y, f, f_para, d_conv, cov_type, order):
    '''
    Estimate the Css matrix for the collocation (covariance function of the signal
    at the known points)
    '''
    r = np.zeros((len(x), len(x)))
    fee = np.ones((len(x), len(x)))
    fen = np.zeros((len(x), len(x)))
    fne = np.zeros((len(x), len(x)))
    fnn = np.ones((len(x), len(x)))
    if d_conv == 'sphere':
        for i, j in itertools.product(xrange(len(x)), xrange(len(x))):
            r[i,j] = g.inv(x[i], y[i], x[j], y[j])[2] / 1000.
            fee[i,j] = calc_fee(x[i], y[i], x[j], y[j])
            fne[i,j] = calc_fne(x[i], y[i], x[j], y[j])
            fen[i,j] = calc_fen(x[i], y[i], x[j], y[j])
            fnn[i,j] = calc_fnn(x[i], y[i], x[j], y[j])
    elif d_conv == 'utm':
        for i, j in itertools.product(xrange(len(x)), xrange(len(x))):
            r[i,j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    else:
        sys.exit('ERROR: Wrong choice of distance calculation provided. Only "sphere" and "utm" are valid arguments.')
    if len(f_para) == 2:
        Css = covariance_function(f)[0](r, f_para[0], f_para[1])
    elif len(f_para) == 3:
        Css = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2])
    elif len(f_para) == 4:
        Css = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2], f_para[3])
    if cov_type == 'jl' and d_conv == 'sphere':
        if order == 'EW+EW':
            Css = Css * fee
        elif order == 'EW+NS':
            Css = Css * fen
        elif order == 'NS+EW':
            Css = Css * fne
        elif order == 'NS+NS':
            Css = Css * fnn
    print 'Signal-covariance matrix Css created'
    return Css;
