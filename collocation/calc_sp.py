#!/usr/bin/python

import numpy as np

def calc_sp(Cps, Czz_m1, l):
    '''
    Calculate sp (signal at unknown points)
    '''
    sp = np.dot(np.dot(Cps, Czz_m1), l)
    print('Signal at unknown points calculated')
    return sp;
