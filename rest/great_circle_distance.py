#!/usr/bin/python

import numpy as np

def great_circle_distance(lon1, lat1, lon2, lat2):
    '''
    Calculate central angle between two points using great-circle distance
    Formula from https://en.wikipedia.org/wiki/Great-circle_distance
    '''
    if lon1 == lon2 and lat1 == lat2:
        angle = 0
    else:
        lat1r = np.radians(lat1)
        lat2r = np.radians(lat2)
        lon12r = np.radians(abs(lon1 - lon2))
        angle = np.arccos((np.sin(lat1r) * np.sin(lat2r)) + (np.cos(lat1r) * np.cos(lat2r) * np.cos(lon12r)))
    return np.degrees(angle);
