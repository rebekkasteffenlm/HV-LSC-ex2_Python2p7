#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
from rest import *


def plot_new_stations(data, data_new, plate, plate_loc, name, map):
    '''
    Plot new stations and old stations with different colors for different plates
    '''
    color_list = [['darkgreen', 'lightgreen'],
                  ['darkblue', 'blue'],
                  ['darkred', 'red'],
                  ['orange', 'yellow'],
                  ['black', 'grey'],
                  ['saddlebrown', 'chocolate'],
                  ['darkorchid', 'orchid'],
                  ['magenta', 'pink'],
                  ['dodgerblue', 'skyblue'],
                  ['darkcyan', 'cyan'],
                  ['olive', 'khaki']]
    
    k = 0
    for i in xrange(0, len(plate)):
        if i >= len(color_list):
            k = 0
        plate_num = plate.keys()[i]
        locs = np.where(plate_loc[:,2] == plate_num)[0]
        x, y = map(data[locs,0], data[locs,1])
        map.scatter(x, y, s=10, c=color_list[k][0], edgecolors='black', linewidths=0.25, zorder=2)
        x, y = map(data_new[locs,0], data_new[locs,1])
        map.scatter(x, y, s=10, c=color_list[k][1], edgecolors='black', linewidths=0.25, zorder=3)
        k += 1
    plt.savefig('figures/' + str(name) + '_stations.png', dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    plt.clf()
    print 'Station plot %s_stations.png created'%(name)
    return;
