#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
from geopy.distance import geodesic
from rest.isfloat import isfloat

def plotting_area(lon, lat, bound_para, pb_model=[0]):
    '''
    Install plotting area
    '''
    lon0 = np.round((np.nanmax(lon) - np.nanmin(lon)) / 2. + np.nanmin(lon), 1)
    lat0 = np.round((np.nanmax(lat) - np.nanmin(lat)) / 2. + np.nanmin(lat), 1)
    a = np.where(np.logical_and(lat >= lat0 - 5, lat <= lat0 + 5))[0]
    b = np.where(np.logical_and(lon >= lon0 - 10, lon <= lon0 + 10))[0]
    w = geodesic((lat0, np.nanmin(lon)), (lat0, np.nanmax(lon))).km
    w = np.ceil(w / 10**(len(str(w).split('.')[0]) - 2)) * 10**(len(str(w).split('.')[0]) - 2)
    w_add = np.ceil(w / 10**(len(str(w).split('.')[0]) - 1)) * 10**(len(str(w).split('.')[0]) - 2)
    h = geodesic((np.nanmin(lat), lon0), (np.nanmax(lat), lon0)).km
    h = np.ceil(h / 10**(len(str(h).split('.')[0]) - 2)) * 10**(len(str(h).split('.')[0]) - 2)
    h_add = np.ceil(h / 10**(len(str(h).split('.')[0]) - 1)) * 10**(len(str(h).split('.')[0]) - 2)
    dlon = np.round(abs(np.ceil(np.nanmax(lon)) - np.floor(np.nanmin(lon))) / 5., 0)
    dlat = np.round(abs(np.ceil(np.nanmax(lat)) - np.floor(np.nanmin(lat))) / 5., 0)
    if dlat == 0:
        dlat = 1
    elif 0 < dlat <= 3:
        dlat = 2
    elif 3 < dlat <= 7:
        dlat = 5
    elif 7 < dlat <= 12:
        dlat = 10
    elif 12 < dlat < 25:
        dlat = 20
    elif dlat >= 25:
        dlat = 30
    if dlon == 0:
        dlon = 1
    elif 0 < dlon <= 3:
        dlon = 2
    elif 3 < dlon <= 7:
        dlon = 5
    elif 7 < dlon <= 12:
        dlon = 10
    elif 12 < dlon < 25:
        dlon = 20
    elif dlon >= 25:
        dlon = 30
    
    if w_add > 400:
        lon_add = 5
    else:
        lon_add = 0
    
    if h_add > 400:
        lat_add = 2
    else:
        lat_add = 0
    
    lon_add = 0

    if (np.round(w / h, 1) == 0) or (np.round(h / w, 1) == 0):
        m = Basemap(projection='eck4',lon_0=0,resolution='i')
        m.drawmapboundary(linewidth=2)
        m.drawmeridians(np.arange(-180,180,45), labels=[0,0,1,1], linewidth=0.3, dashes=(None,None), fontsize=7)
        m.drawparallels(np.arange(-90,90,30), labels=[1,1,0,0], linewidth=0.3, dashes=(None,None), fontsize=7)
    else:
        m = Basemap(projection='aeqd', resolution='i', lon_0=lon0+lon_add, lat_0=lat0+lat_add, width=(w + 3 * w_add) * 1000., height=(h + 2 * h_add) * 1000., area_thresh=1000)
        m.drawmapboundary(linewidth=2)
        m.drawmeridians(np.arange(-170,170,dlon), labels=[0,0,0,1], linewidth=0.3, dashes=(None,None), fontsize=9)
        m.drawparallels(np.arange(-90,90,dlat), labels=[1,1,0,0], linewidth=0.3, dashes=(None,None), fontsize=9)
    if bound_para == 'y':
        try:
            m.drawcoastlines(linewidth=0.3)
        except:
            pass
        try:
            m.drawcountries(linewidth=0.3, linestyle='solid', color='darkblue')
        except:
            pass
    if len(pb_model) > 1:
        file = open('data/plate_boundaries/' + pb_model[0])
        boundaries = file.readlines()
        file.close()
        num_bounds = []
        for i in xrange(len(boundaries)):
            if isfloat(boundaries[i].split(',')[0]) is False:
                num_bounds.append(i)
        for i in xrange(0, len(num_bounds) / 2):
            bound = []
            for j in xrange(len(boundaries)):
                if num_bounds[i * 2] < j < num_bounds[(i * 2) + 1]:
                    if isfloat(boundaries[j].split(',')[0]) is True:
                        lon = boundaries[j].split()[0].split(',')[pb_model[1] - 1]
                        lat = boundaries[j].split()[0].split(',')[pb_model[2] - 1]
                        bound.append([float(lon), float(lat)])
            bound = np.array(bound)
            locs = np.where(bound[:,0] < 0)[0]
            locs_inv = np.where(np.logical_and(bound[:,0] >= 0, bound[:,0] <= 180))[0]
            if len(locs) > 1:
                loc_steps = np.where(np.diff(locs) != 1)[0]
                if len(loc_steps) >= 1:
                    k = 0
                    for j in xrange(len(loc_steps)):
                        x, y = m(bound[locs[k:loc_steps[j]+1],0], bound[locs[k:loc_steps[j]+1],1])
                        m.plot(x, y, color='gold', linewidth=2.0)
                        k = loc_steps[j] + 1
                    x, y = m(bound[locs[k:],0], bound[locs[k:],1])
                    m.plot(x, y, color='gold', linewidth=2.0)
                else:
                    x, y = m(bound[locs,0], bound[locs,1])
                    m.plot(x, y, color='gold', linewidth=2.0)
            if len(locs_inv) > 1:
                loc_steps = np.where(np.diff(locs_inv) != 1)[0]
                if len(loc_steps) >= 1:
                    k = 0
                    for j in xrange(len(loc_steps)):
                        x, y = m(bound[locs_inv[k:loc_steps[j]+1],0], bound[locs_inv[k:loc_steps[j]+1],1])
                        m.plot(x, y, color='gold', linewidth=2.0)
                        k = loc_steps[j] + 1
                    x, y = m(bound[locs_inv[k:],0], bound[locs_inv[k:],1])
                    m.plot(x, y, color='gold', linewidth=2.0)
                else:
                    x, y = m(bound[locs_inv,0], bound[locs_inv,1])
                    m.plot(x, y, color='gold', linewidth=2.0)
    
    return m, (w + 3 * w_add), (h + 2 * h_add);
