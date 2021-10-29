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

def get_cps(x, y, xi, yi, f, f_para, d_conv, cov_type, order):
    '''
    Estimate the Cps matrix for the collocation (covariance function of the signal
    at the known points to the unknown points)
    '''
    r = np.zeros((len(xi), len(x)))
    fee = np.ones((len(xi), len(x)))
    fen = np.zeros((len(xi), len(x)))
    fne = np.zeros((len(xi), len(x)))
    fnn = np.ones((len(xi), len(x)))
    if d_conv == 'sphere':
        for i, j in itertools.product(xrange(len(xi)), xrange(len(x))):
            r[i,j] = g.inv(xi[i], yi[i], x[j], y[j])[2] / 1000.
            fee[i,j] = calc_fee(xi[i], yi[i], x[j], y[j])
            fne[i,j] = calc_fne(xi[i], yi[i], x[j], y[j])
            fen[i,j] = calc_fen(xi[i], yi[i], x[j], y[j])
            fnn[i,j] = calc_fnn(xi[i], yi[i], x[j], y[j])
    elif d_conv == 'utm':
        for i, j in itertools.product(xrange(len(xi)), xrange(len(x))):
            r[i,j] = np.sqrt((xi[i] - x[j])**2 + (yi[i] - y[j])**2)
    else:
        sys.exit('ERROR: Wrong choice of distance calculation provided. Only "sphere" and "utm" are valid arguments.')
    if len(f_para) == 2:
        Cps = covariance_function(f)[0](r, f_para[0], f_para[1])
    elif len(f_para) == 3:
        Cps = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2])
    elif len(f_para) == 4:
        Cps = covariance_function(f)[0](r, f_para[0], f_para[1], f_para[2], f_para[3])
    if cov_type == 'jl' and d_conv == 'sphere':
        if order == 'EW+EW':
            Cps = Cps * fee
        elif order == 'EW+NS':
            Cps = Cps * fen
        elif order == 'NS+EW':
            Cps = Cps * fne
        elif order == 'NS+NS':
            Cps = Cps * fnn
    print 'Covariance matrix Cps created'
    return Cps;
