#!/usr/bin/python

import numpy as np

def calc_dsp(Cps, Cpp, Czz_m1):
    '''
    Calculate error covariance matrix of the signal sp and the mean error dsp
    '''
    sigmasp = Cpp - np.dot(np.dot(Cps, Czz_m1), Cps.T)
    dsp = np.sqrt(np.abs(np.diag(sigmasp)))
    print('Uncertainty of signal at unknown points calculated')
    return dsp;
