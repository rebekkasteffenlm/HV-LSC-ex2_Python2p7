#!/usr/bin/python

import numpy as np
import itertools
import sys
import get_covariance_parameter
from function_covariance import *
from plotting import *
from rest import *
from pyproj import Geod
R = 6371000
g = Geod(ellps='WGS84')
from scipy.stats import pearsonr


def empirical_covariance(x, y, data1, data2, noise1, noise2, delta, d_conv, f, filenamei, nr):
    '''
    Calculate empirical covariance of the input data to find signal covariance and correlation length
    Calculations are based on the parameter delta, which defines the discrete interval in km or degree
    '''
    
    #Outlier removal
    rms_data1 = 3 * np.sqrt(np.sum(data1**2) / float(len(data1)))
    if len(np.where(abs(data1) > rms_data1)[0]) > 0:
        data1 = data1[np.where(abs(data1) <= rms_data1)[0]]
        x = x[np.where(abs(data1) <= rms_data1)[0]]
        y = y[np.where(abs(data1) <= rms_data1)[0]]
        data2 = data2[np.where(abs(data1) <= rms_data1)[0]]
        noise1 = noise1[np.where(abs(data1) <= rms_data1)[0]]
        noise2 = noise2[np.where(abs(data1) <= rms_data1)[0]]
        print('Outlier identified and removed')
    rms_data2 = 3 * np.sqrt(np.sum(data2**2) / float(len(data2)))
    if len(np.where(abs(data2) > rms_data2)[0]) > 0:
        data2 = data2[np.where(abs(data2) <= rms_data2)[0]]
        x = x[np.where(abs(data2) <= rms_data2)[0]]
        y = y[np.where(abs(data2) <= rms_data2)[0]]
        data1 = data1[np.where(abs(data2) <= rms_data2)[0]]
        noise1 = noise1[np.where(abs(data2) <= rms_data2)[0]]
        noise2 = noise2[np.where(abs(data2) <= rms_data2)[0]]
        print('Outlier identified and removed')
    
    c01 = (np.sum(data1**2) / len(data1)) - (np.sum(noise1**2) / len(noise1))
    c02 = (np.sum(data2**2) / len(data2)) - (np.sum(noise2**2) / len(noise2))
    c0 = c01 * 0.5 + c02 * 0.5
    if d_conv == 'sphere':
        dmax = np.degrees(g.inv(np.amin(x), np.amin(y), np.amax(x), np.amax(y))[2] / float(R))
        c = []
        b1 = []; b2 = []
        c.append([0, c0, np.std(np.c_[data1, data2]), len(data1)])
        for k in xrange(1, 100):
            a = []
            if k == 1:
                delta_min = 0
                delta_max = delta
            else:
                delta_min = (2 * k - 3) * delta
                delta_max = (2 * k - 1) * delta
            for i, j in itertools.product(xrange(len(data1)), xrange(len(data2))):
                if i != j:
                    dij = np.degrees(g.inv(x[i], y[i], x[j], y[j])[2] / float(R))
                    if delta_min < dij <= delta_max:
                        a.append(data1[i] * data2[j])
            if (k > 30) and (len(a) < np.round(len(data1), -1 * (len(str(len(data1)))- 1)) * 5):
                break
            if (k <= 30) and (len(a) <= (len(data1) * 2.)):
                continue
            if len(a) != 0:
                c.append([(delta_min + delta_max) / 2., np.sum(a) / float(len(a) - 1), np.std(a), len(a)])
                if (np.sum(a) / float(len(a))) <= (c[0][1] / 2.):
                    b1.append(delta_min)
                    b2.append(delta_max)
        c = np.asarray(c)
        cova = c[:,1]
        cova_dist = np.radians(c[:,0]) * R / 1000.
        dij = []
        for i, j in itertools.product(xrange(len(data1)), xrange(len(data2))):
            if i != j:
                dij.append(np.degrees(g.inv(x[i], y[i], x[j], y[j])[2] / float(R)))
            else:
                dij.append(0)
        mean_dij = np.mean(dij)
        if len(b1) > 0:
            corr_bounds = np.radians(np.array([b1[0], b2[0]])) * R / 1000.
        else:
            corr_bounds = np.array([mean_dij, cova_dist[-1]])
        print 'Mean distance of this dataset is %.4f km / %.3f degrees'%(np.radians(mean_dij) * R / 1000., mean_dij)
    elif d_conv == 'utm':
        dmax = np.sqrt((np.amax(x) - np.amin(x))**2 + (np.amax(y) - np.amin(y))**2)
        c = []
        b1 = []; b2 = []
        c.append([0, c0, np.std(np.c_[data1, data2]), len(data1)])
        for k in xrange(1, 100):
            a = []
            if k == 1:
                delta_min = 0
                delta_max = delta
            else:
                delta_min = (2 * k - 3) * delta
                delta_max = (2 * k - 1) * delta
            for i, j in itertools.product(xrange(len(data1)), xrange(len(data2))):
                if i != j:
                    dij = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if delta_min < dij <= delta_max:
                        a.append(data1[i] * data2[j])
            if (k > 30) and (len(a) < np.round(len(data1), -1 * (len(str(len(data1)))- 1)) * 5):
                break
            if (k <= 30) and (len(a) < len(data1)):
                continue
            if len(a) != 0:
                c.append([(delta_min + delta_max) / 2., np.sum(a) / float(len(a) - 1), np.std(a), len(a)])
                if (np.sum(a) / float(len(a))) <= (c[0][1] / 2.):
                    b1.append(delta_min)
                    b2.append(delta_max)
        c = np.asarray(c)
        cova = c[:,1]
        cova_dist = c[:,0]
        dij = []
        for i, j in itertools.product(xrange(len(data1)), xrange(len(data2))):
            dij.append(np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2))
        mean_dij = np.mean(dij)
        if len(b1) > 0:
            corr_bounds = np.array([b1[0], b2[0]])
        else:
            corr_bounds = np.array([mean_dij, cova_dist[-1]])
        print 'Mean distance of this dataset is %.4f km'%(mean_dij)
    else:
        sys.exit('ERROR: Wrong choice of distance calculation provided. Only "sphere" and "utm" are valid arguments.')
    
    
    popt, perr, filename = get_covariance_parameter.get_covariance_parameter(cova, cova_dist, corr_bounds, covariance_function(f)[0], f, covariance_function(f)[1], filenamei)
    if len(popt) == 2:
        curve = covariance_function(f)[0](cova_dist, popt[0], popt[1])
    elif len(popt) == 3:
        curve = covariance_function(f)[0](cova_dist, popt[0], popt[1], popt[2])
    elif len(popt) == 4:
        curve = covariance_function(f)[0](cova_dist, popt[0], popt[1], popt[2], popt[3])
    plot_covariance(f, covariance_function(f)[1], popt, cova, cova_dist, c[:,2], curve[:], filename + '_' + str(nr))
    for i in xrange(len(cova)):
        print '%.2f %.4f %.4f %d %.4f %.4e'%(c[i,0], c[i,1], c[i,2], c[i,3], curve[i], curve[i] - c[i,1])
    misfit = np.sqrt(sum((curve - c[:,1])**2) / float(len(c))) / c[0,1]
    misfit_1st = np.sqrt(sum((curve[:3] - c[:3,1])**2) / float(len(c[:3]))) / c[0,1]
    print('Misfit is: %.5f'%(misfit))
    print('Misfit of the first three points is: %.5f'%(misfit_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(c[:,1], curve)[0]))
    
    corr_bounds = np.array([cova_dist[0], cova_dist[-1]])
    popt1, perr1, filename1 = get_covariance_parameter.get_covariance_parameter(cova, cova_dist, corr_bounds, covariance_function(f)[0], f, covariance_function(f)[1], filenamei)
    if len(popt1) == 2:
        curve1 = covariance_function(f)[0](cova_dist, popt1[0], popt1[1])
    elif len(popt1) == 3:
        curve1 = covariance_function(f)[0](cova_dist, popt1[0], popt1[1], popt1[2])
    elif len(popt1) == 4:
        curve1 = covariance_function(f)[0](cova_dist, popt1[0], popt1[1], popt1[2], popt1[3])
    plot_covariance(f, covariance_function(f)[1], popt1, cova, cova_dist, c[:,2], curve1[:], filename1 + '_' + str(nr))
    for i in xrange(len(cova)):
        print '%.2f %.4f %.4f %d %.4f %.4e'%(c[i,0], c[i,1], c[i,2], c[i,3], curve1[i], curve1[i] - c[i,1])
    misfit1 = np.sqrt(sum((curve1 - c[:,1])**2) / float(len(c))) / c[0,1]
    misfit1_1st = np.sqrt(sum((curve1[:3] - c[:3,1])**2) / float(len(c[:3]))) / c[0,1]
    print('Misfit is: %.5f'%(misfit1))
    print('Misfit of the first three points is: %.5f'%(misfit1_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(c[:,1], curve1)[0]))
    
    if (misfit1 <= misfit) and (misfit1_1st <= misfit_1st):
        popt = popt1 + [0]
        perr = perr1 + [0]
        curve = curve1 + [0]
        filename = filename1 + ''
        misfit = misfit1 + 0
        misfit_1st = misfit1_1st + 0
        del popt1; del perr1,; del filename1
    elif (misfit1 > misfit) and (misfit1_1st <= misfit_1st) and np.round(pearsonr(c[:,1], curve1)[0], 2) >= np.round(pearsonr(c[:,1], curve)[0], 2):
        if popt1[1] < popt[1]:
            popt = popt1 + [0]
            perr = perr1 + [0]
            curve = curve1 + [0]
            filename = filename1 + ''
            misfit = misfit1 + 0
            misfit_1st = misfit1_1st + 0
            del popt1; del perr1,; del filename1
    elif (misfit1 <= misfit) and (misfit1_1st > misfit_1st) and np.round(pearsonr(c[:,1], curve1)[0], 2) >= np.round(pearsonr(c[:,1], curve)[0], 2):
        if popt1[1] < popt[1]:
            popt = popt1 + [0]
            perr = perr1 + [0]
            curve = curve1 + [0]
            filename = filename1 + ''
            misfit = misfit1 + 0
            misfit_1st = misfit1_1st + 0
            del popt1; del perr1,; del filename1
    
    print('Final values:')
    print('The following parameters for the covariance function ' + covariance_function(f)[1] + ' are obtained:')
    if f in ['gm1', 'gm2', 'reilly', 'hirvonen', 'markov1', 'markov2', 'tri', 'lauer']:
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d'%(popt[0], perr[0], popt[1], perr[1]))
    elif f == 'vestol':
        print('C0 = %.5f +/- %.5f, a = %.3e +/- %.3e, b = %.3e +/- %.3e'%(popt[0], perr[0], popt[1], perr[1], popt[2], perr[2]))
    elif f == 'gauss':
        print('C0 = %.5f +/- %.5f, alpha = %.3e +/- %.3e'%(popt[0], perr[0], popt[1], perr[1]))
    elif f == 'log':
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d, m = %.3f +/- %.3f'%(popt[0], perr[0], popt[1], perr[1], popt[2], perr[2]))
    print('Misfit is: %.5f'%(misfit))
    print('Misfit of the first three points is: %.5f'%(misfit_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(c[:,1], curve)[0]))
    headerline = 'Distance Empirical-Cov Std Number-of-velocities Cov-function'
    np.savetxt('results/' + filename + '_' + str(nr) + '.dat', np.c_[c[:,0], c[:,1], c[:,2:], curve[:]], fmt='%.2f %.4f %.4f %d %.4f', header=headerline)
    print('File %s_%s.dat in results/ created'%(filename, nr))
    return popt, perr;
