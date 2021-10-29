#!/usr/bin/python

import numpy as np

def calc_fen(lon1, lat1, lon2, lat2):
    '''
    Calculate fen component
    '''
    if lon1 == lon2:
        fen = 0
    else:
        lat1r = np.radians(lat1)
        lon12r = np.radians(lon1 - lon2)
        fen = np.sin(lon12r) * np.sin(lat1r)
    return fen;
