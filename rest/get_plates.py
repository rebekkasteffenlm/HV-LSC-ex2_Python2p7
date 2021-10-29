#!/usr/bin/python

import numpy as np
from plotting import *

def get_plates(pb_model, lon_mb, lat_mb, plot_para):
    filef = open('data/plate_boundaries/' + pb_model[1])
    boundaries = filef.readlines()
    filef.close()
    map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1])
    num_bounds = []
    for i in xrange(len(boundaries)):
        if isfloat(boundaries[i].split(',')[0]) is False:
            num_bounds.append(i)
    boundary = {}
    for i in xrange(0, len(num_bounds) / 2):
        bound = []
        for j in xrange(len(boundaries)):
            if num_bounds[i * 2] < j < num_bounds[(i * 2) + 1]:
                if isfloat(boundaries[j].split(',')[0]) is True:
                    lo = boundaries[j].split()[0].split(',')[pb_model[2] - 1]
                    la = boundaries[j].split()[0].split(',')[pb_model[3] - 1]
                    xx, yy = map(float(lo), float(la))
                    bound.append([xx, yy])
        bound = np.array(bound)
        is_in_map = 'n'
        for j in bound:
            xx = j[0]; yy = j[1]
            if (0 <= xx <= w * 1000.) and (0 <= yy <= h * 1000.):
                is_in_map = 'y'
                break
        if is_in_map == 'y':
            boundary[boundaries[num_bounds[i * 2]]] = bound
    
    filef = open('data/plate_boundaries/' + pb_model[0])
    plates = filef.readlines()
    filef.close()
    num_plates = []
    for i in xrange(len(plates)):
        if isfloat(plates[i].split(',')[0]) is False:
            num_plates.append(i)
    plate = {}
    for i in xrange(0, len(num_plates) / 2):
        pla = []
        for j in xrange(len(plates)):
            if num_plates[i * 2] < j < num_plates[(i * 2) + 1]:
                if isfloat(plates[j].split(',')[0]) is True:
                    lo = plates[j].split()[0].split(',')[pb_model[2] - 1]
                    la = plates[j].split()[0].split(',')[pb_model[3] - 1]
                    xx, yy = map(float(lo), float(la))
                    pla.append([xx, yy])
        pla = np.array(pla)
        is_in_map = 'n'
        for j in pla:
            xx = j[0]; yy = j[1]
            if (0 <= xx <= w * 1000.) and (0 <= yy <= h * 1000.):
                is_in_map = 'y'
                break
        if is_in_map == 'y':
            plate[plates[num_plates[i * 2]]] = pla
    
    plate_bounds = []
    for i in xrange(len(plate.keys()) - 1):
        for j in xrange(i + 1, len(plate.keys())):
            for k in xrange(len(boundary.keys())):
                if (plate.keys()[i].split()[0] in boundary.keys()[k]) and (plate.keys()[j].split()[0] in boundary.keys()[k]):
                    plate_bounds.append([k, plate.keys()[i].split()[0], plate.keys()[j].split()[0]])
    
    plate_bounds_name = ['-'.join(row[1:]) for row in plate_bounds]
    plate_bounds_name = np.unique(plate_bounds_name)
    plate_bounds = np.array(plate_bounds)
    boundary_new = {}
    for i in plate_bounds_name:
        i1 = i.split('-')[0]
        i2 = i.split('-')[1]
        ind = np.where(np.logical_or(np.logical_and(plate_bounds[:,1] == i1, plate_bounds[:,2] == i2), np.logical_and(plate_bounds[:,2] == i1, plate_bounds[:,1] == i2)))[0]
        if len(ind) > 1:
            a = np.empty((0, 2))
            for j in ind:
                a = np.concatenate([a, boundary.values()[int(plate_bounds[j,0])]])
            if (np.amax(a[:,0]) - np.amin(a[:,0])) >= (np.amax(a[:,1]) - np.amin(a[:,1])):
                b = a[np.argsort(a, axis=0)[:,0],:]
            elif (np.amax(a[:,0]) - np.amin(a[:,0])) < (np.amax(a[:,1]) - np.amin(a[:,1])):
                b = a[np.argsort(a, axis=0)[:,1],:]
        else:
            b = boundary.values()[int(plate_bounds[ind[0],0])]
        boundary_new[i] = b
    plt.clf()
    return plate, boundary_new, map;
