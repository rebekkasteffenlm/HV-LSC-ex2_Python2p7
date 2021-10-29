#!/usr/bin/python

import numpy as np
import subprocess
import os

def cross_validation(num, data, l, n, l_coll, string):
    '''
    Save data without one station and run collocation --> interpolate at missing station
    '''
    lon_new = np.hstack((data[0:num,0], data[num+1:len(data),0]))
    lon_1 = data[num,0]
    lat_new = np.hstack((data[0:num,1], data[num+1:len(data),1]))
    lat_1 = data[num,1]
    l_new = np.vstack((l[0:num,:], l[num+1:len(l),:]))
    if len(n) > 1:
        n_new = np.vstack((n[0:num,:], n[num+1:len(n),:]))
        n_1 = n[num,:]
        np.savetxt('data/station_test_all_%d.dat'%(num+1), np.c_[lon_new, lat_new, l_new, n_new])
    else:
        np.savetxt('data/station_test_all_%d.dat'%(num+1), np.c_[lon_new, lat_new, l_new])
    np.savetxt('data/station_test_%d.dat'%(num+1), np.c_[lon_1, lat_1])
    string = string + ' -fi station_test_all_%d.dat -lon 1 -lat 2 -obs'%(num+1)
    for j in xrange(3, l.shape[1] + 3):
        string = string + ' %d'%(j)
    j += 1
    if len(n) > 1:
        string = string + ' -err'
        for k in xrange(j, n.shape[1] + j):
            string = string + ' %d'%(k)
    string = string + ' -p txt data/station_test_%d.dat 1 2'%(num+1)
    
    subprocess.check_call(string + ' > cross-validation_out_%d.dat'%(num+1), shell=True)
    
    file = open('cross-validation_out_%d.dat'%(num+1), 'r')
    a = file.readlines()
    file.close()
    folder = a[-1].split()[-1]
    files = os.listdir('results/' + folder + '/')
    for j in files:
        if '_collocation_np-txt_mean_bg.dat' in j:
            res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
            break
    if 'res' not in locals():
        for j in files:
            if '_collocation_np-txt_mean.dat' in j:
                res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
                break
    if 'res' not in locals():
        for j in files:
            if '_collocation_np-txt.dat' in j:
                res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
                break
    
    l_1 = l[num,:]
    l_2 = []
    for j in xrange(0, len(l_1)):
        l_2.append(res[j + 2])
    l_2 = np.asarray(l_2)
    diff1 = l_1 - l_2
    l_coll_1 = l_coll[num,:]
    diff2 = l_coll_1 - l_2
    
    subprocess.check_call('rm data/station_test_%d.dat data/station_test_all_%d.dat'%(num+1, num+1), shell=True)
    subprocess.check_call('rm -r results/%s figures/%s'%(folder, folder), shell=True)
    subprocess.check_call('rm cross-validation_out_%d.dat'%(num+1), shell=True)
        
    return np.hstack((num+1, lon_1, lat_1, l_1[:], l_coll_1[:], l_2[:], diff1[:], diff2[:]));

