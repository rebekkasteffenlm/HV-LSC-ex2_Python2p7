#!/usr/bin/python

import numpy as np

def calc_ds(Css, Czz_m1):
    '''
    Calculate error covariance matrix of the signal s and the mean error ds
    '''
    sigmas = Css - np.dot(np.dot(Css, Czz_m1), Css)
    ds = np.sqrt(np.abs(np.diag(sigmas)))
    print('Uncertainty of signal at known points calculated')
    return ds;
