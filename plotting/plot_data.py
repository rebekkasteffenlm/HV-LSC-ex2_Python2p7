#!/usr/bin/python

from plotting_area import *
from plot_points import *
from plot_contourf import *
from plot_vectors import *

def plot_data(lon, lat, data, filename, plot_para, clabel_order, pb_plot=[0], map_lon=[0], map_lat=[0], comp_data=[0], bound_lon=[0], bound_lat=[0]):
    '''
    Plot data
    '''
    if len(map_lon) == 1:
        map_lon = lon
    if len(map_lat) == 1:
        map_lat = lat
    if len(bound_lon) == 1:
        bound_lon = map_lon
    if len(bound_lat) == 1:
        bound_lat = map_lat
    if data.shape[1] > 1:
        for i in xrange(0, data.shape[1]):
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            clabel = clabel_order[i][0] + '-' + clabel_order[i][1]
            fig_filename = plot_points(np.c_[lon, lat, data[:,i]], filename + '_' + clabel_order[i], clabel + ' ' + plot_para[2], plot_para[0], map)
            print 'Point plot %s created'%(fig_filename)
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            fig_filename = plot_contourf(lon, lat, data[:,i], filename + '_' + clabel_order[i], clabel + ' ' + plot_para[2], plot_para[0], map)
            print 'Filled contour plot %s created'%(fig_filename)
        if plot_para[0] == 'bwr':
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
            fig_filename = plot_points(np.c_[lon, lat, np.sqrt(data[:,clabel_order.index('EW')]**2 + data[:,clabel_order.index('NS')]**2)], filename + clabels, plot_para[2], 'viridis', map)
            print 'Point plot %s created'%(fig_filename)
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            fig_filename = plot_contourf(lon, lat, np.sqrt(data[:,clabel_order.index('EW')]**2 + data[:,clabel_order.index('NS')]**2), filename + clabels, plot_para[2], 'viridis', map)
            print 'Filled contour plot %s created'%(fig_filename)
        else:
            clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            fig_filename = plot_points(np.c_[lon, lat, np.sqrt(data[:,clabel_order.index('EW')]**2 + data[:,clabel_order.index('NS')]**2)], filename + clabels, plot_para[2], plot_para[0], map)
            print 'Point plot %s created'%(fig_filename)
            map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
            fig_filename = plot_contourf(lon, lat, np.sqrt(data[:,clabel_order.index('EW')]**2 + data[:,clabel_order.index('NS')]**2), filename + clabels, plot_para[2], plot_para[0], map)
            print 'Filled contour plot %s created'%(fig_filename)
    else:
        map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
        fig_filename = plot_points(np.c_[lon, lat, data[:]], filename + '_' + '+'.join(clabel_order), plot_para[2], plot_para[0], map)
        print 'Point plot %s created'%(fig_filename)
        map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
        fig_filename = plot_contourf(lon, lat, data[:], filename + '_' + '+'.join(clabel_order), plot_para[2], plot_para[0], map)
        print 'Filled contour plot %s created'%(fig_filename)
    if data.shape[1] > 1:
        clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
        fig,ax = plt.subplots()
        map, w, h = plotting_area(bound_lon, bound_lat, plot_para[1], pb_plot)
        fig_filename, b = plot_vectors(np.c_[lon, lat, data[:,clabel_order.index('EW')], data[:,clabel_order.index('NS')]], filename + clabels, plot_para[3], map)
        print 'Vector plot %s created'%(fig_filename)
        if len(comp_data) > 1:
            fig_filename, b = plot_vectors(np.c_[map_lon, map_lat, comp_data[:,clabel_order.index('EW')], comp_data[:,clabel_order.index('NS')]], filename + clabels + '_comp', plot_para[3], map, 'red', b[0], b[1])
            print 'Vector plot %s created'%(fig_filename)
            plt.clf()
    plt.clf()
    return;
