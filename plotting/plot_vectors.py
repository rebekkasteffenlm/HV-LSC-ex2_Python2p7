#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
from matplotlib.patches import Polygon
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def plot_vectors(data, name, clabel, m, co='black', scalef=0, length_arrow=0, qwidth=0):
    '''
    Plot vectors of the data
    '''
    ew, ns, x, y = m.rotate_vector(data[:,2], data[:,3], data[:,0], data[:,1], returnxy=True)
    if len(np.where(np.logical_and(x == np.round(m.xmax / 2., 4), y == np.round(m.ymax / 2., 4)))[0]) > 0:
        locs = np.where(np.logical_and(x == np.round(m.xmax / 2., 4), y == np.round(m.ymax / 2., 4)))[0]
        for i in locs:
            ew[i], ns[i], xs, ys = m.rotate_vector(data[i,2], data[i,3], data[i,0]+1e-4, data[i,1]+1e-4, returnxy=True)
            del xs; del ys
    if scalef == 0:
        scalef_y = (ns.max() - ns.min()) / (y.max() - y.min())
        scalef_x = (ew.max() - ew.min()) / (x.max() - x.min())
        scalef = ((scalef_y + scalef_x) / 2.)
        area = ((y.max() - y.min()) * (x.max() - x.min())) / 1e12
        if area > 10:
            scalef = scalef * 5.
        else:
            scalef = scalef * 10.
    
    if length_arrow == 0:
        a = [0.01, 0.05, 0.1, 0.5, 1,
             2, 2.5, 3, 4, 5, 6, 7, 7.5, 8, 9, 10,
             12.5, 15, 17.5, 20, 25, 30, 40, 50,
             60, 70, 75, 80, 90, 100]
        max_length = np.sqrt(ew**2 + ns**2).max()
        for i in range(len(a)):
            if max_length > a[i]:
                length_arrow = a[i]
    
    if qwidth != 0:
        q = m.quiver(x, y, ew, ns, color=co, angles='xy', scale_units='xy', scale=scalef, pivot='tail', width=qwidth, zorder=50, headwidth=4)
    else:
        q = m.quiver(x, y, ew, ns, color=co, angles='xy', scale_units='xy', scale=scalef, pivot='tail', zorder=50, headwidth=4)
    
    if co == 'black':
        qk = plt.quiverkey(q, .87, .95, length_arrow, str(length_arrow) + ' ' + clabel, labelpos='S', labelsep=0.05, color='black', fontproperties={'weight': 'bold', 'size': 12}, zorder=50)
    else:
        qk = plt.quiverkey(q, .87, .88, length_arrow, str(length_arrow) + ' ' + clabel, labelpos='S', labelsep=0.05, color='red', fontproperties={'weight': 'bold', 'size': 12}, zorder=50)
    
    q._init()
    filename = str(name) + '_vectors.png'
    plt.savefig('figures/' + str(filename), dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    return filename, [q.scale, length_arrow, q.width];
