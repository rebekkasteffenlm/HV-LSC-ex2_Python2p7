#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *

def plot_points(data, name, cb_label, cscale, m, z_min=0, z_max=0, step=0):
    '''
    Plot magnitude of the data as points with specific colours
    '''
    if z_min == 0 and z_max == 0 and step == 0:
        data_min = np.nanmin(data[:,2])
        data_max = np.nanmax(data[:,2])
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
    
    x, y = m(data[:,0], data[:,1])
    plotcf = m.scatter(x, y, c=data[:,2], s=30, edgecolors='none', vmin=z_min, vmax=z_max, cmap=plt.get_cmap(cscale))
    cb = m.colorbar(plotcf, location='bottom', size='3%', pad='5%')
    cb.ax.tick_params(labelsize=9)
    cb.set_label(cb_label, fontsize=10)
    filename = str(name) + '_co-points.png'
    plt.savefig('figures/' + str(filename), dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    plt.clf()
    return filename;
