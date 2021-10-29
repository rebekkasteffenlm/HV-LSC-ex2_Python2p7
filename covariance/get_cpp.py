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

def get_cpp(xi, yi, f, f_para, d_conv, cov_type, order):
    '''
    Estimate the Cpp matrix for the collocation (covariance function of the signal
    at the unknown points)
    '''
    r = np.zeros((len(xi), len(xi)))
    fee = np.ones((len(xi), len(xi)))
    fen = np.zeros((len(xi), len(xi)))
    fne = np.zeros((len(xi), len(xi)))
    fnn = np.ones((len(xi), len(xi)))
    if d_conv == 'sphere':
        for i, j in itertools.product(xrange(len(xi)), xrange(len(xi))):
            r[i,j] = g.inv(xi[i], yi[i], xi[j], yi[j])[2] / 1000.
            fee[i,j] = calc_fee(xi[i], yi[i], xi[j], yi[j])
            fne[i,j] = calc_fne(xi[i], yi[i], xi[j], yi[j])
            fen[i,j] = calc_fen(xi[i], yi[i], xi[j], yi[j])
            fnn[i,j] = calc_fnn(xi[i], yi[i], xi[j], yi[j])
    elif d_conv == 'utm':
        for i, j in itertools.product(xrange(len(xi)), xrange(len(xi))):
            r[i,j] = np.sqrt((xi[i] - xi[j])**2 + (yi[i] - yi[j])**2)
    else:
        sys.exit('ERROR: Wrong choice of distance calculation provided. Only "sphere" and "utm" are valid arguments.')
    if len(f_para) == 2:
        Cpp = covariance_function(f)[0](r, f_para[0], f_para[1])
    elif len(f_para) == 3:
        Cpp = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2])
    elif len(f_para) == 4:
        Cpp = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2], f_para[3])
    if cov_type == 'jl' and d_conv == 'sphere':
        if order == 'EW+EW':
            Cpp = Cpp * fee
        elif order == 'EW+NS':
            Cpp = Cpp * fen
        elif order == 'NS+EW':
            Cpp = Cpp * fne
        elif order == 'NS+NS':
            Cpp = Cpp * fnn
    print 'Covariance matrix Cpp created'
    return Cpp;
