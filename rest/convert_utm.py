#!/usr/bin/python

import numpy as np
import pyproj    
import utm

def convert_utm(lon, lat, zone_num):
    '''
    Transform longitude and latitude datasets into UTM coordinates (in km)
    '''
    myProj = pyproj.Proj(proj='utm', zone=zone_num, ellps='WGS84', datum='WGS84')
    x, y = myProj(lon, lat)
    return np.asarray(x) / 1000., np.asarray(y) / 1000.;
