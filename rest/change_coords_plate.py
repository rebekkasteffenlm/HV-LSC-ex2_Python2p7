#!/usr/bin/python

import numpy as np
import shapely.geometry as geom
from pyproj import Geod
from scipy.spatial import ConvexHull
R = 6371000
g = Geod(ellps='WGS84')

def change_coords_plate(lon, lat, plate, boundary, dist, map, loni=[0], lati=[0]):
    '''
    Change coordinates due to known plate boundaries
    '''
    
    x_map, y_map = map(lon, lat)
    x_map_mod = x_map + [0]; y_map_mod = y_map + [0]
    plate_loc = []
    for j in xrange(0, len(x_map)):
        k = 0
        for i in xrange(0, len(plate)):
            if geom.Point(x_map[j], y_map[j]).within(geom.Polygon(plate.values()[i])) is True:
                plate_loc.append([x_map[j], y_map[j], plate.keys()[i]])
                k += 1
        if k == 0:
            for i in xrange(0, len(plate)):
                if np.round(geom.Point(x_map[j], y_map[j]).distance(geom.Polygon(plate.values()[i])), 1) == 0.:
                    plate_loc.append([x_map[j], y_map[j], plate.keys()[i]])
                    break
    plate_loc = np.asarray(plate_loc)
    
    if len(loni) > 1:
        xi_map, yi_map = map(loni, lati)
        xi_map_mod = xi_map + [0]; yi_map_mod = yi_map + [0]
        plate_loci = []
        for j in xrange(0, len(xi_map)):
            k = 0
            for i in xrange(0, len(plate)):
                if geom.Point(xi_map[j], yi_map[j]).within(geom.Polygon(plate.values()[i])) is True:
                    plate_loci.append([xi_map[j], yi_map[j], plate.keys()[i]])
                    k += 1
            if k == 0:
                for i in xrange(0, len(plate)):
                    if np.round(geom.Point(xi_map[j], yi_map[j]).distance(geom.Polygon(plate.values()[i])), 1) == 0.:
                        plate_loci.append([xi_map[j], yi_map[j], plate.keys()[i]])
                        break
        plate_loci = np.asarray(plate_loci)
    else:
        plate_loci = np.empty((0, 3))
        xi_map = np.empty((0, 1))
        yi_map = np.empty((0, 1))
    
    x_xi_map = np.hstack((x_map, xi_map))
    y_yi_map = np.hstack((y_map, yi_map))
    plate_loc_loci = np.vstack((plate_loc, plate_loci))
    x_min = x_xi_map.min()
    x_max = x_xi_map.max()
    y_min = y_yi_map.min()
    y_max = y_yi_map.max()
    x_mean = (x_max - x_min) / 2.
    y_mean = (y_max - y_min) / 2.
    map_polygon = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    central_plate = []
    for i in xrange(0, len(plate)):
        k = 0
        for j in plate.values()[i]:
            if geom.Point(j[0], j[1]).within(geom.Polygon(map_polygon)) is True:
                k += 1
            else:
                break
        if k == len(plate.values()[i]):
            lon_mean1, lat_mean1 = map(x_mean, y_mean, inverse=True)
            lon_mean2, lat_mean2 = map(np.mean(plate.values()[i][:,0]), np.mean(plate.values()[i][:,1]), inverse=True)
            central_plate.append([plate.keys()[i], np.mean(plate.values()[i][:,0]), np.mean(plate.values()[i][:,1]), g.inv(lon_mean1, lat_mean1, lon_mean2, lat_mean2)[2], lon_mean2, lat_mean2])
    if len(central_plate) == 0:
        nums = []
        for i in xrange(0, len(plate)):
            nums.append(len(np.where(plate_loc_loci[:,2] == plate.keys()[i])[0]))
        nums = np.array(nums)
        loc = np.argsort(nums)[0]
        lon_mean1, lat_mean1 = map(x_mean, y_mean, inverse=True)
        lon_mean2, lat_mean2 = map(np.mean(plate.values()[loc][:,0]), np.mean(plate.values()[loc][:,1]), inverse=True)
        central_plate = [plate.keys()[loc], np.mean(plate.values()[loc][:,0]), np.mean(plate.values()[loc][:,1]), g.inv(lon_mean1, lat_mean1, lon_mean2, lat_mean2)[2], lon_mean2, lat_mean2]
    elif len(central_plate) == 1:
        central_plate = central_plate[0]
    else:
        central_plate = np.array(central_plate)
        loc = np.argsort(central_plate[:,3].astype(float))[0]
        central_plate = central_plate[loc]
    
    x_xi_map_mod = x_xi_map + [0]
    y_yi_map_mod = y_yi_map + [0]
    for i in xrange(0, len(plate)):
        print('Working on plate %s'%(plate.keys()[i].split()[0]))
        if plate.keys()[i] == central_plate[0]:
            continue
        plate_locs = np.where(plate_loc_loci[:,2] == plate.keys()[i])[0]
        plate_x_mean = np.mean(x_xi_map[plate_locs])
        plate_y_mean = np.mean(y_yi_map[plate_locs])
        plate_lon_mean, plate_lat_mean = map(plate_x_mean, plate_y_mean, inverse=True)
        az12, az21, geodesic_dist = g.inv(central_plate[4], central_plate[5], plate_lon_mean, plate_lat_mean)
        lon0_mod, lat0_mod, backaz = g.fwd(plate_lon_mean, plate_lat_mean, az12, dist * 1000.)
        for j in plate_locs:
            lon_map, lat_map = map(x_xi_map[j], y_yi_map[j], inverse=True)
            az12, az21, geodesic_dist = g.inv(plate_lon_mean, plate_lat_mean, lon_map, lat_map)
            endlon, endlat, backaz = g.fwd(lon0_mod, lat0_mod, az12, geodesic_dist)
            new_x, new_y = map(endlon, endlat)
            x_xi_map_mod[j] = new_x
            y_yi_map_mod[j] = new_y
    
    plate_hulls = {}
    for i in xrange(0, len(plate)):
        plate_locs = np.where(plate_loc_loci[:,2] == plate.keys()[i])[0]
        hull = ConvexHull(np.c_[x_xi_map_mod[plate_locs], y_yi_map_mod[plate_locs]])
        plate_hulls[plate.keys()[i]] = np.c_[x_xi_map_mod[plate_locs], y_yi_map_mod[plate_locs]][hull.vertices]
    plate_problems = []
    for i in xrange(len(plate_hulls)):
        plate_name = plate_hulls.keys()[i]
        for j in [a for a in xrange(len(plate_hulls.keys())) if plate_hulls.keys()[a] not in plate_hulls.keys()[i]]:
            for k in plate_hulls.values()[j]:
                for l in plate_hulls.values()[i]:
                    lon1, lat1 = map(k[0], k[1], inverse=True)
                    lon2, lat2 = map(l[0], l[1], inverse=True)
                    if g.inv(lon1, lat1, lon2, lat2)[2] < dist * 1000.:
                        plate_problems.append([plate_name, plate_hulls.keys()[j]])
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    
    while len(plate_problems) > 0:
        i1 = plate.keys().index(plate_problems[0][0])
        i2 = plate.keys().index(plate_problems[0][1])
        if plate_problems[0][0] == central_plate[0]:
            plate_fix = i1; plate_loose = i2
        elif plate_problems[0][1] == central_plate[0]:
            plate_fix = i2; plate_loose = i1
        else:
            lon_mean2, lat_mean2 = map(np.mean(plate.values()[i1][:,0]), np.mean(plate.values()[i1][:,1]), inverse=True)
            dist1 = g.inv(lon_mean1, lat_mean1, lon_mean2, lat_mean2)[2]
            lon_mean2, lat_mean2 = map(np.mean(plate.values()[i2][:,0]), np.mean(plate.values()[i2][:,1]), inverse=True)
            dist2 = g.inv(lon_mean1, lat_mean1, lon_mean2, lat_mean2)[2]
            if dist1 < dist2:
                plate_fix = i1; plate_loose = i2
            else:
                plate_fix = i2; plate_loose = i1
                
        print('Working on plate %s again'%(plate.keys()[plate_loose].split()[0]))
        plate_locs = np.where(plate_loc_loci[:,2] == plate.keys()[plate_loose])[0]
        plate_x_mean = np.mean(x_xi_map_mod[plate_locs])
        plate_y_mean = np.mean(y_yi_map_mod[plate_locs])
        plate_fix_locs = np.where(plate_loc_loci[:,2] == plate.keys()[plate_fix])[0]
        plate_fix_x_mean = np.mean(x_xi_map_mod[plate_fix_locs])
        plate_fix_y_mean = np.mean(y_yi_map_mod[plate_fix_locs])
        plate_lon_mean, plate_lat_mean = map(plate_x_mean, plate_y_mean, inverse=True)
        plate_fix_lon, plate_fix_lat = map(plate_fix_x_mean, plate_fix_y_mean, inverse=True)
        az12, az21, geodesic_dist = g.inv(plate_fix_lon, plate_fix_lat, plate_lon_mean, plate_lat_mean)
        lon0_mod, lat0_mod, backaz = g.fwd(plate_lon_mean, plate_lat_mean, az12, dist * 1000.)
        if lat0_mod >= 85:
            lon0_mod, lat0_mod, backaz = g.fwd(plate_lon_mean, plate_lat_mean, az12-90, dist * 1000.)
        for j in plate_locs:
            lon_map, lat_map = map(x_xi_map_mod[j], y_yi_map_mod[j], inverse=True)
            az12, az21, geodesic_dist = g.inv(plate_lon_mean, plate_lat_mean, lon_map, lat_map)
            endlon, endlat, backaz = g.fwd(lon0_mod, lat0_mod, az12, geodesic_dist)
            new_x, new_y = map(endlon, endlat)
            x_xi_map_mod[j] = new_x
            y_yi_map_mod[j] = new_y
                
        plate_hulls = {}
        for i in xrange(0, len(plate)):
            plate_locs = np.where(plate_loc_loci[:,2] == plate.keys()[i])[0]
            hull = ConvexHull(np.c_[x_xi_map_mod[plate_locs], y_yi_map_mod[plate_locs]])
            plate_hulls[plate.keys()[i]] = np.c_[x_xi_map_mod[plate_locs], y_yi_map_mod[plate_locs]][hull.vertices]

        plate_problems = []
        for i in xrange(len(plate_hulls)):
            plate_name = plate_hulls.keys()[i]
            for j in [a for a in xrange(len(plate_hulls.keys())) if plate_hulls.keys()[a] not in plate_hulls.keys()[i]]:
                for k in plate_hulls.values()[j]:
                    for l in plate_hulls.values()[i]:
                        lon1, lat1 = map(k[0], k[1], inverse=True)
                        lon2, lat2 = map(l[0], l[1], inverse=True)
                        if g.inv(lon1, lat1, lon2, lat2)[2] < dist * 1000.:
                            plate_problems.append([plate_name, plate_hulls.keys()[j]])
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
    
    lon_mod, lat_mod = map(x_xi_map_mod, y_yi_map_mod, inverse=True)
    return lon_mod, lat_mod, plate_loc_loci;
