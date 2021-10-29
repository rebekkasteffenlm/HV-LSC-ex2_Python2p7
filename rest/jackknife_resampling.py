#!/usr/bin/python

import numpy as np
import subprocess
import os

def jackknife_resampling(nums, data, datai, l, n, string):
    '''
    Save data without one station and run collocation --> interpolate at missing station
    '''
    lon_new = np.hstack((data[0:nums[0],0], data[nums[0]+1:len(data),0]))
    lat_new = np.hstack((data[0:nums[0],1], data[nums[0]+1:len(data),1]))
    l_new = np.vstack((l[0:nums[0],:], l[nums[0]+1:len(l),:]))
    if len(n) > 1:
        n_new = np.vstack((n[0:nums[0],:], n[nums[0]+1:len(n),:]))
        n_1 = n[nums[0],:]
        np.savetxt('data/station_test_all_%d_%d.dat'%(nums[0]+1, nums[1]+1), np.c_[lon_new, lat_new, l_new, n_new])
    else:
        np.savetxt('data/station_test_all_%d_%d.dat'%(nums[0]+1, nums[1]+1), np.c_[lon_new, lat_new, l_new])
    np.savetxt('data/station_test_%d_%d.dat'%(nums[0]+1, nums[1]+1), datai)
    string = string + ' -fi station_test_all_%d_%d.dat -lon 1 -lat 2 -obs'%(nums[0]+1, nums[1]+1)
    for j in xrange(3, l.shape[1] + 3):
        string = string + ' %d'%(j)
    j += 1
    if len(n) > 1:
        string = string + ' -err'
        for k in xrange(j, n.shape[1] + j):
            string = string + ' %d'%(k)
    string = string + ' -p txt data/station_test_%d_%d.dat 1 2'%(nums[0]+1, nums[1]+1)
    
    subprocess.check_call(string + ' > cross-validation_out_%d_%d.dat'%(nums[0]+1, nums[1]+1), shell=True)
    
    file = open('cross-validation_out_%d_%d.dat'%(nums[0]+1, nums[1]+1), 'r')
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
    
    l_2 = []
    for j in xrange(l.shape[1]):
        l_2.append(res[j + 2])
    l_2 = np.asarray(l_2)
    
    subprocess.check_call('rm data/station_test_%d_%d.dat data/station_test_all_%d_%d.dat'%(nums[0]+1, nums[1]+1, nums[0]+1, nums[1]+1), shell=True)
    subprocess.check_call('rm -r results/%s figures/%s'%(folder, folder), shell=True)
    subprocess.check_call('rm cross-validation_out_%d_%d.dat'%(nums[0]+1, nums[1]+1), shell=True)
    
    return l_2;

