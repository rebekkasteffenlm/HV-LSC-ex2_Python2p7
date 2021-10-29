#!/usr/bin/python

import numpy as np

def calc_n(Cnn, Czz_m1, l):
    '''
    Calculate n (noise)
    '''
    n = np.dot(np.dot(Cnn, Czz_m1), l)
    print('Noise at known points calculated')
    return n;
