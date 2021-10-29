#!/usr/bin/python

import numpy as np

def calc_fne(lon1, lat1, lon2, lat2):
    '''
    Calculate fne component
    '''
    if lon1 == lon2:
        fne = 0
    else:
        lat2r = np.radians(lat2)
        lon21r = np.radians(lon2 - lon1)
        fne = np.sin(lon21r) * np.sin(lat2r)
    return fne;
