#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *

def plot_contourf(lon, lat, data, name, cb_label, cscale, m):
    '''
    Plot filled contours of the data
    '''
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_max_min = data_max - data_min
    steps = np.asarray([1, 2, 2.5, 5, 7.5])
    for i in xrange(1,4):
        steps = np.concatenate([steps, steps * 10**i, steps / 10**i])
    steps = np.unique(steps)
    for i in steps:
        if data_max_min <= (i * 15):
            step = i
            break
    i = 0
    while (i * step) <= data_max:
        z_max = i * step
        i += 1
    z_max = i * step
    i = 0
    if data_min >= 0:
        while (i * step) <= data_min:
            z_min = i * step
            i += 1
    else:
        while (i * step) > data_min:
            z_min = i * step
            i += -1
        z_min = i * step
    if cscale == 'bwr':
        if abs(z_min) >= abs(z_max):
            if z_min < 0 and z_max > 0:
                z_max = abs(z_min)
            else:
                z_max = 0
        elif abs(z_min) < abs(z_max):
            if z_min < 0 and z_max > 0:
                z_min = -1 * abs(z_max)
            else:
                z_min = 0
    levels = np.arange(float(z_min),(float(z_max) + float(step)),float(step))
    
    x, y = m(lon, lat)
    plotcf = m.contourf(x, y, data, levels, vmin=z_min, vmax=z_max, cmap=plt.get_cmap(cscale), tri=True)
    cb = m.colorbar(plotcf, location='bottom', size='3%', pad='5%')
    cb.ax.tick_params(labelsize=7)
    cb.set_label(cb_label, fontsize=7)
    filename = str(name) + '_contour.png'
    plt.savefig('figures/' + str(filename), dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    plt.clf()
    return filename;
