#!/usr/bin/python

import numpy as np

def calc_s(Css, Czz_m1, l):
    '''
    Calculate s (signals)
    '''
    s = np.dot(np.dot(Css, Czz_m1), l)
    print('Signal at given points calculated')
    return s;
