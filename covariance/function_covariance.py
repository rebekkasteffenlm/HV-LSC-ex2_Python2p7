#!/usr/bin/python

import numpy as np
import sys

def covariance_function(function_name):
    '''
    Definition of the covariance functions
    '''
    if function_name == 'gm1':
        function = func_gm1
        function_name_long = 'Gauss-Markov 1st order'
    elif function_name == 'gm2':
        function = func_gm2
        function_name_long = 'Gauss-Markov 2nd order'
    elif function_name == 'reilly':
        function = func_reilly
        function_name_long = 'Reilly model'
    elif function_name == 'hirvonen':
        function = func_hirvonen
        function_name_long = 'Hirvonen\'s formula'
    elif function_name == 'markov1':
        function = func_markov1
        function_name_long = 'Markov 1st order'
    elif function_name == 'markov2':
        function = func_markov2
        function_name_long = 'Markov 2nd order'
    elif function_name == 'tri':
        function = func_tri
        function_name_long = 'Triangular model'
    elif function_name == 'lauer':
        function = func_lauer
        function_name_long = 'Lauer'
    elif function_name == 'vestol':
        function = func_vestol
        function_name_long = 'Vestol'
    elif function_name == 'gauss':
        function = func_gauss
        function_name_long = 'Gauss'
    elif function_name == 'log':
        function = func_log
        function_name_long = 'Logarithmic'
    else:
        sys.exit('ERROR: Chosen function is not supported.')
    return function, function_name_long;


def func_gauss(dist, C0, alpha):
    '''
    Gaussian function with the factor alpha
    '''
    return C0 * np.exp(-1 * alpha**2 * dist**2);


def func_gm1(dist, C0, d0):
    '''
    First-order Gauss-Markov process
    '''
    return C0 * np.exp(-1 * dist / d0);


def func_gm2(dist, C0, d0):
    '''
    Second-order Gauss-Markov process
    '''
    return C0 * np.exp(-1 * dist**2 / d0**2);


def func_reilly(dist, C0, d0):
    '''
    Reilly covariance function
    '''
    return C0 * (1 - 0.5 * (dist / d0)**2) * np.exp(-0.5 * (dist / d0)**2);


def func_hirvonen(dist, C0, d0):
    '''
    Hirvonen covariance function (usually used for grvaity data)
    '''
    return C0 * (d0**2 / (d0**2 + dist**2));


def func_log(dist, C0, d0, m):
    '''
    Logarithmic covariance function (m=2 is Hirvonen function)
    '''
    return C0 * (d0**m / (d0**m + dist**m));


def func_markov1(dist, C0, d0):
    '''
    First-order Markov covariance function
    '''
    return C0 * (1 + (dist / d0)) * np.exp(-1 * dist / d0);


def func_markov2(dist, C0, d0):
    '''
    Second-order Markov covariance function
    '''
    return C0 * (1 + (dist / d0) + (dist**2 / (3 * d0**2))) * np.exp(-1 * dist / d0);


def func_vestol(dist, C0, a, b):
    '''
    Covariance function used by Olav Vestol with a=10/400^2 and b=8/400
    '''
    return C0 * ((a * dist**2) + (b * dist) + 1);


def func_tri(dist, C0, d0):
    '''
    Triangular covariance function
    '''
    return C0 * (1 - (dist / (2 * d0)));


def func_lauer(dist, C0, d0):
    '''
    Lauer covariance function
    '''
    return C0 / (dist**d0);
