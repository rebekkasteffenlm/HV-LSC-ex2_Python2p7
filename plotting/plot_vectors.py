#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
from matplotlib.patches import Polygon
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def plot_vectors(data, name, clabel, m, co='black', b=0, factor=0):
    '''
    Plot vectors of the data
    '''
    ew, ns, x, y = m.rotate_vector(data[:,2], data[:,3], data[:,0], data[:,1], returnxy=True)
    if len(np.where(np.logical_and(x == np.round(m.xmax / 2., 4), y == np.round(m.ymax / 2., 4)))[0]) > 0:
        locs = np.where(np.logical_and(x == np.round(m.xmax / 2., 4), y == np.round(m.ymax / 2., 4)))[0]
        for i in locs:
            ew[i], ns[i], xs, ys = m.rotate_vector(data[i,2], data[i,3], data[i,0]+1e-4, data[i,1]+1e-4, returnxy=True)
            del xs; del ys
    if np.nanmax(ew) >= abs(np.nanmin(ew)):
        length_ew = abs(np.nanmax(ew))
    elif np.nanmax(ew) < abs(np.nanmin(ew)):
        length_ew = abs(np.nanmin(ew))
    if np.nanmax(ns) >= abs(np.nanmin(ns)):
        length_ns = abs(np.nanmax(ns))
    elif np.nanmax(ns) < abs(np.nanmin(ns)):
        length_ns = abs(np.nanmin(ns))
    if length_ew >= length_ns:
        length = length_ew
    elif length_ew < length_ns:
        length = length_ns
    if factor == 0:
        if np.round(length, -1) > 20:
            factor = 1.
        elif 10 <= np.round(length, -1) <= 20:
            factor = 2.
        else:
            factor = 3.
        if length < 1:
            factor = 5.
    q = m.quiver(x, y, ew * factor, ns * factor, color=co, units='width', scale=3e-5, scale_units='xy', pivot='tail', angles='xy', zorder=50, width=0.0035)
    a = [0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 50, 75, 100]
    if b == 0:
        b = 1
        for i in a:
            if i > length:
                break
            b = i
    if co == 'black':
        qk = plt.quiverkey(q, .87, .95, 0.01, str(b) + ' ' + clabel, labelpos='S', labelsep=0.05, color='black', fontproperties={'weight': 'bold', 'size': 12}, zorder=50)
    else:
        qk = plt.quiverkey(q, .87, .88, 0.01, str(b) + ' ' + clabel, labelpos='S', labelsep=0.05, color='red', fontproperties={'weight': 'bold', 'size': 12}, zorder=50)
    filename = str(name) + '_vectors.png'
    plt.savefig('figures/' + str(filename), dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    return filename, [b,factor];
