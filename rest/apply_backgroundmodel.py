#!/usr/bin/python

import numpy as np
from scipy import interpolate
from convert_utm import *
from in_hull import *

def apply_backgroundmodel(bg_model, obs, distance_conv, x, y):
    filef = np.loadtxt('data/' + bg_model[0], dtype='S')
    lonbg = filef[:, int(bg_model[1]) - 1].astype(float)
    latbg = filef[:, int(bg_model[2]) - 1].astype(float)
    model = np.empty((len(lonbg), 0))
    for i in bg_model[3]:
        model = np.c_[model, filef[:, int(i) - 1].astype(float)]
    obs_corrected = obs + 0
    if distance_conv == 'utm':
        xn = x + 0
        yn = y + 0
    elif distance_conv == 'sphere':
        zone_num = utm.from_latlon(np.mean(x), np.mean(y))[2]
        xn, yn = convert_utm(x, y, zone_num)
    xm, ym = convert_utm(lonbg, latbg, zone_num)
    modelxy = np.empty((len(x), 0))
    for i in xrange(model.shape[1]):
        model_xy = interpolate.griddata((xm, ym), model[:,i], (xn, yn), method='cubic')
        for j in xrange(len(model_xy)):
            if in_hull(np.c_[xm, ym], np.c_[xn[j], yn[j]]) is False:
                model_xy[j] = 0.
        if len(np.argwhere(np.isnan(model_xy))) > 0:
            fspline = interpolate.SmoothBivariateSpline(xn[np.isfinite(model_xy)], yn[np.isfinite(model_xy)],
                                                        model_xy[np.isfinite(model_xy)], kx=5, ky=5)
            for j in np.argwhere(np.isnan(model_xy)):
                model_xy[j] = fspline(xn[j], yn[j])[0][0]
        obs_corrected[:,i] = obs_corrected[:,i] - model_xy[:]
        modelxy = np.c_[modelxy, model_xy]
    return lonbg, latbg, model, obs_corrected, modelxy;
