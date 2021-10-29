#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *


def plot_stations(data, name, m):
    '''
    Plot distribution of stations
    '''
    x, y = m(np.array(data[:,0]), np.array(data[:,1]))
    if len(data) > 6200:
        m.scatter(x, y, s=6, c='darkgreen', edgecolors='black', linewidths=0.25, zorder=3)
    else:
        m.scatter(x, y, s=10, c='darkgreen', edgecolors='black', linewidths=0.25, zorder=3)
    plt.savefig('figures/' + str(name) + '_stations.png', dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    plt.clf()
    print 'Station plot %s_stations.png created'%(name)
    return;
