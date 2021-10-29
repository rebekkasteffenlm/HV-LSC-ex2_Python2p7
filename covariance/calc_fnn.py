#!/usr/bin/python

import numpy as np

def calc_fnn(lon1, lat1, lon2, lat2):
    '''
    Calculate fnn component
    '''
    lon12r = np.radians(lon1 - lon2)
    fnn = np.cos(lon12r)
    return fnn;