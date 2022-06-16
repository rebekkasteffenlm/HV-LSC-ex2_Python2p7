#!/usr/bin/python

import sys
import os
import itertools
import numpy as np
from calc_fee import *
from calc_fen import *
from calc_fne import *
from calc_fnn import *
from function_covariance import *
import get_covariance_parameter
from plotting import *
from rest import *
from pyproj import Geod
R = 6371000
g = Geod(ellps='WGS84')
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import multiprocessing

def type_function(type1, type2):
    '''
    '''
    if type1 == 'e' and type2 == 'e':
        type_f = calc_fee
    elif type1 == 'e' and type2 == 'n':
        type_f = calc_fen
    elif type1 == 'n' and type2 == 'e':
        type_f = calc_fne
    elif type1 == 'n' and type2 == 'n':
        type_f = calc_fnn
    return type_f;


def calc_beta_std(x1, x2, y1, y2, l1, l2, n1, n2, f_function):
    '''
    Calculate beta (equation (83))
    '''
    f = f_function(x1, y1, x2, y2)
    beta = (l1 * l2 - np.cov(np.c_[n1, n2])) / float(f)
    return beta;


def calc_cov_beta4(xi, yi, noise, fu, f_para, value_set, j, k):
    '''
    Calculate covariance of beta (equation (86))
    '''
    pt1 = int(value_set[j][0])
    pt2 = int(value_set[j][1])
    pt3 = int(value_set[k][0])
    pt4 = int(value_set[k][1])
    n1 = noise[pt1][int(value_set[j][2])]
    n2 = noise[pt2][int(value_set[j][3])]
    n3 = noise[pt3][int(value_set[k][2])]
    n4 = noise[pt4][int(value_set[k][3])]
    type_s = value_set[j][4]
    type_t = value_set[j][5]
    type_u = value_set[k][4]
    type_v = value_set[k][5]
    f_function = [type_function(type_s, type_t), type_function(type_s, type_u),
                  type_function(type_s, type_v), type_function(type_t, type_u),
                  type_function(type_t, type_v), type_function(type_u, type_v)]
    x = xi[[pt1, pt2, pt3, pt4]]
    y = yi[[pt1, pt2, pt3, pt4]]
    
    x1 = x[0]; x2 = x[1]; x3 = x[2]; x4 = x[3]
    y1 = y[0]; y2 = y[1]; y3 = y[2]; y4 = y[3]
    fst = f_function[0](x1, y1, x2, y2)
    fsu = f_function[1](x1, y1, x3, y3)
    fsv = f_function[2](x1, y1, x4, y4)
    ftu = f_function[3](x2, y2, x3, y3)
    ftv = f_function[4](x2, y2, x4, y4)
    fuv = f_function[5](x3, y3, x4, y4)
    
    rsu = g.inv(x1, y1, x3, y3)[2] / 1000.
    rsv = g.inv(x1, y1, x4, y4)[2] / 1000.
    rtu = g.inv(x2, y2, x3, y3)[2] / 1000.
    rtv = g.inv(x2, y2, x4, y4)[2] / 1000.
    if len(f_para) == 2:
        Ksu = covariance_function(fu)[0](rsu, f_para[0], f_para[1])
        Ksv = covariance_function(fu)[0](rsv, f_para[0], f_para[1])
        Ktu = covariance_function(fu)[0](rtu, f_para[0], f_para[1])
        Ktv = covariance_function(fu)[0](rtv, f_para[0], f_para[1])
    elif len(f_para) == 3:
        Ksu = covariance_function(fu)[0](rsu, f_para[0], f_para[1], f_para[2])
        Ksv = covariance_function(fu)[0](rsv, f_para[0], f_para[1], f_para[2])
        Ktu = covariance_function(fu)[0](rtu, f_para[0], f_para[1], f_para[2])
        Ktv = covariance_function(fu)[0](rtv, f_para[0], f_para[1], f_para[2])
    elif len(f_para) == 4:
        Ksu = covariance_function(fu)[0](rsu, f_para[0], f_para[1], f_para[2], f_para[3])
        Ksv = covariance_function(fu)[0](rsv, f_para[0], f_para[1], f_para[2], f_para[3])
        Ktu = covariance_function(fu)[0](rtu, f_para[0], f_para[1], f_para[2], f_para[3])
        Ktv = covariance_function(fu)[0](rtv, f_para[0], f_para[1], f_para[2], f_para[3])
    
    Csu = np.cov(np.c_[n1, n3])
    Csv = np.cov(np.c_[n1, n4])
    Ctu = np.cov(np.c_[n2, n3])
    Ctv = np.cov(np.c_[n2, n4])
    
    cov_beta = ((fsu * ftv * Ksu * Ktv) + (fsv * ftu * Ksv * Ktu) + (ftv * Ktv * Csu) + (fsv * Ksv * Ctu)
                + (ftu * Ktu * Csv) + (ftu * Ktu * Csv) + (fsu * Ksu * Ctv) + (Csu * Ctv) + (Csv * Ctu)) / (fst * fuv)
    return cov_beta;


def get_covariance_function4_std(x, y, data, noise, delta, fu, f_para, filenamei, ncore):
    '''
    Calculate empirical covariance of the input data following Juliette Legrand (2007)
    '''
    
    #Outlier removal
    rms_data1 = 3 * np.sqrt(np.sum(data[:,0]**2) / float(len(data)))
    rms_data2 = 3 * np.sqrt(np.sum(data[:,1]**2) / float(len(data)))
    if len(np.where(abs(data[:,0]) > rms_data1)[0]) > 0:
        data[:,0] = data[np.where(abs(data[:,0]) <= rms_data1)[0],0]
        x = x[np.where(abs(data[:,0]) <= rms_data1)[0]]
        y = y[np.where(abs(data[:,0]) <= rms_data1)[0]]
        data[:,1] = data[np.where(abs(data[:,0]) <= rms_data1)[0],1]
        noise[:,0] = noise[np.where(abs(data[:,0]) <= rms_data1)[0],0]
        noise[:,1] = noise[np.where(abs(data[:,0]) <= rms_data1)[0],1]
        print('Outlier identified and removed')
    if len(np.where(abs(data[:,1]) > rms_data2)[0]) > 0:
        data[:,1] = data[np.where(abs(data[:,1]) <= rms_data2)[0],1]
        x = x[np.where(abs(data[:,1]) <= rms_data2)[0]]
        y = y[np.where(abs(data[:,1]) <= rms_data2)[0]]
        data[:,0] = data[np.where(abs(data[:,1]) <= rms_data2)[0],0]
        noise[:,0] = noise[np.where(abs(data[:,1]) <= rms_data2)[0],0]
        noise[:,1] = noise[np.where(abs(data[:,1]) <= rms_data2)[0],1]
        print('Outlier identified and removed')
    
    dmax = np.degrees(g.inv(np.amin(x), np.amin(y), np.amax(x), np.amax(y))[2] / float(R))
    ddata_dict = {}
    ddata_dict[0, 0] = np.c_[np.linspace(0, len(x) - 1, len(x)).astype(int), np.linspace(0, len(x) - 1, len(x)).astype(int)]
    for k in xrange(1, int(np.ceil(dmax / delta))):
        a = []
        if k == 1:
            delta_min = 0
            delta_max = delta
        else:
            delta_min = (2 * k - 3) * delta
            delta_max = (2 * k - 1) * delta
        for i, j in itertools.product(xrange(len(x)), xrange(len(x))):
            if i != j:
                dij = np.degrees(g.inv(x[i], y[i], x[j], y[j])[2] / float(R))
                if delta_min < dij <= delta_max:
                    a.append([i, j])
        if (k > 30) and (len(a) < np.round(len(data), -1 * (len(str(len(data))) - 1)) * 5):
            break
        if (k <= 30) and (len(a) <= (len(x) * 2.)):
            continue
        if len(a) > 30:
            a = np.asarray(a)
            indices = np.argsort(a, axis=1)
            b = np.array([a[i,indices[i]] for i in xrange(len(indices))])
            try:
                a = np.unique(b, axis=0)
            except:
                a = []
            if len(a) == 0:
                try:
                    a = np.vstack({tuple(row) for row in b})
                except:
                    a = []
            if len(a) == 0:
                sys.exit('Problems with finding unique rows in NumPy (https://getridbug.com/python/find-unique-rows-in-numpy-array/)')
            ddata_dict[delta_min, delta_max] = a
        
    li_dict = {}
    for i in xrange(len(ddata_dict.keys())):
        li = []
        for j in xrange(len(ddata_dict.values()[i])):
            pt1 = ddata_dict.values()[i][j][0]
            pt2 = ddata_dict.values()[i][j][1]
            li.append([pt1, pt2, 0, 0, 'e', 'e'])
            li.append([pt1, pt2, 1, 1, 'n', 'n'])
            if (pt1 != pt2) and (float('%.3f'%(x[pt1])) != float('%.3f'%(x[pt2]))):
                li.append([pt1, pt2, 0, 1, 'e', 'n'])
                li.append([pt1, pt2, 1, 0, 'n', 'e'])
        li_dict[ddata_dict.keys()[i]] = np.asarray(li)
        
    emp_K = []; emp_varK = []
    for i in xrange(len(li_dict.keys())):
        beta = np.zeros((len(li_dict.values()[i]), 1))
        mean_e = np.mean(data[np.unique(li_dict.values()[i][:,0].astype(int)),0])
        mean_n = np.mean(data[np.unique(li_dict.values()[i][:,1].astype(int)),1])
        std_e = np.std(data[np.unique(li_dict.values()[i][:,0].astype(int)),0])
        std_n = np.std(data[np.unique(li_dict.values()[i][:,1].astype(int)),1])
        for j in xrange(len(li_dict.values()[i])):
            pt1 = int(li_dict.values()[i][j][0])
            pt2 = int(li_dict.values()[i][j][1])
            type_k = li_dict.values()[i][j][4]
            type_l = li_dict.values()[i][j][5]
            k = int(li_dict.values()[i][j][2])
            m = int(li_dict.values()[i][j][3])
            if type_k == 'e':
                k = 0
                mean_k = mean_e
            elif type_k == 'n':
                k = 1
                mean_k = mean_n
            if type_l == 'e':
                m = 0
                mean_m = mean_e
            elif type_l == 'n':
                m = 1
                mean_m = mean_e
            beta[j, 0] = calc_beta_std(x[pt1], x[pt2], y[pt1], y[pt2], (data[pt1][k] - mean_k), (data[pt2][m] - mean_m), noise[pt1][k], noise[pt2][m], type_function(type_k, type_l))
        
        a, b = np.meshgrid(xrange(len(li_dict.values()[i])), xrange(len(li_dict.values()[i])))
        ab = zip(a.ravel(), b.ravel())
        cov_beta = Parallel(n_jobs=ncore)(delayed(calc_cov_beta4)(x, y, noise, fu, f_para, li_dict.values()[i], j[0], j[1]) for j in ab)
        cov_beta_m = np.reshape(cov_beta, (len(li_dict.values()[i]), len(li_dict.values()[i])))
        
        cov_beta_i = np.linalg.inv(cov_beta_m * np.identity(cov_beta_m.shape[0]))
        del cov_beta; del cov_beta_m
        id_m = np.ones((len(beta), 1))
        alpha = np.dot(cov_beta_i, id_m) / np.dot(np.dot(id_m.T, cov_beta_i.T), id_m)
        K_h = np.dot(alpha.T, beta) / float(std_e * std_n)
        var_K_h = 1. / np.dot(np.dot(id_m.T, cov_beta_i.T), id_m)
        emp_K.append(K_h[0][0])
        emp_varK.append(np.sqrt(var_K_h[0][0]))
        print(li_dict.keys()[i], K_h[0][0], var_K_h[0][0], np.round(np.sum(alpha), 3))
    
    dij = []
    for i, j in itertools.product(xrange(len(x)), xrange(len(x))):
        if i != j:
            dij.append(np.degrees(g.inv(x[i], y[i], x[j], y[j])[2] / float(R)))
        else:
            dij.append(0)
    mean_dij = np.mean(dij)
    
    cova_dist = np.array([(i[0] + i[1]) / 2. for i in li_dict.keys()])
    b = np.argsort(cova_dist)
    cova_dist_d = cova_dist[b]
    cova_dist = np.radians(cova_dist_d) * R / 1000.
    cova = np.asarray(emp_K)
    cova_error = np.asarray(emp_varK)
    cova_length = np.array([len(i) for i in li_dict.values()])
    cova = cova[b]
    cova_error = cova_error[b]
    cova_length = cova_length[b]
    corr_bounds = np.array([0, np.radians(dmax) * R / 1000.])
    for i in xrange(1, len(cova)):
        if cova[i] < cova[0] / 2.:
            corr_bounds = np.array([cova_dist[i] - np.radians(delta) * R / 1000., cova_dist[i] + np.radians(delta) * R / 1000.])
            break
    
    
    popt, perr, filename = get_covariance_parameter.get_covariance_parameter(cova, cova_dist, corr_bounds, covariance_function(fu)[0], fu, covariance_function(fu)[1], filenamei)
    if len(popt) == 2:
        curve = covariance_function(fu)[0](cova_dist, popt[0], popt[1])
    elif len(popt) == 3:
        curve = covariance_function(fu)[0](cova_dist, popt[0], popt[1], popt[2])
    elif len(popt) == 4:
        curve = covariance_function(fu)[0](cova_dist, popt[0], popt[1], popt[2], popt[3])
    plot_covariance(fu, covariance_function(fu)[1], popt, cova, cova_dist, cova_error, curve[:], filename + '_both')
    for i in xrange(len(cova)):
        print '%.2f %.4f %.4f %d %.4f %.4e'%(cova_dist_d[i], cova[i], cova_error[i], cova_length[i], curve[i], curve[i] - cova[i])
    misfit = np.sqrt(sum((curve - cova)**2) / float(len(cova))) / cova[0]
    misfit_1st = np.sqrt(sum((curve[:3] - cova[:3])**2) / float(len(cova[:3]))) / cova[0]
    print('Misfit is: %.5f'%(misfit))
    print('Misfit of the first three points is: %.5f'%(misfit_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(cova, curve)[0]))
    
    corr_bounds = np.array([cova_dist[0], cova_dist[-1]])
    popt1, perr1, filename1 = get_covariance_parameter.get_covariance_parameter(cova, cova_dist, corr_bounds, covariance_function(fu)[0], fu, covariance_function(fu)[1], filenamei)
    if len(popt1) == 2:
        curve1 = covariance_function(fu)[0](cova_dist, popt1[0], popt1[1])
    elif len(popt1) == 3:
        curve1 = covariance_function(fu)[0](cova_dist, popt1[0], popt1[1], popt1[2])
    elif len(popt1) == 4:
        curve1 = covariance_function(fu)[0](cova_dist, popt1[0], popt1[1], popt1[2], popt1[3])
    plot_covariance(fu, covariance_function(fu)[1], popt, cova, cova_dist, cova_error, curve[:], filename1 + '_both')
    for i in xrange(len(cova)):
        print '%.2f %.4f %.4f %d %.4f %.4e'%(cova_dist_d[i], cova[i], cova_error[i], cova_length[i], curve[i], curve[i] - cova[i])
    misfit1 = np.sqrt(sum((curve1 - cova)**2) / float(len(cova))) / cova[0]
    misfit1_1st = np.sqrt(sum((curve1[:3] - cova[:3])**2) / float(len(cova[:3]))) / cova[0]
    print('Misfit is: %.5f'%(misfit1))
    print('Misfit of the first three points is: %.5f'%(misfit1_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(cova, curve1)[0]))
    
    if (misfit1 <= misfit) and (misfit1_1st <= misfit_1st):
        popt = popt1 + [0]
        perr = perr1 + [0]
        curve = curve1 + [0]
        filename = filename1 + ''
        misfit = misfit1 + 0
        misfit_1st = misfit1_1st + 0
        del popt1; del perr1,; del filename1
    elif (misfit1 > misfit) and (misfit1_1st <= misfit_1st) and np.round(pearsonr(cova, curve1)[0], 2) >= np.round(pearsonr(cova, curve)[0], 2):
        if popt1[1] < popt[1]:
            popt = popt1 + [0]
            perr = perr1 + [0]
            curve = curve1 + [0]
            filename = filename1 + ''
            misfit = misfit1 + 0
            misfit_1st = misfit1_1st + 0
            del popt1; del perr1,; del filename1
    elif (misfit1 <= misfit) and (misfit1_1st > misfit_1st) and np.round(pearsonr(cova, curve1)[0], 2) >= np.round(pearsonr(cova, curve)[0], 2):
        if popt1[1] < popt[1]:
            popt = popt1 + [0]
            perr = perr1 + [0]
            curve = curve1 + [0]
            filename = filename1 + ''
            misfit = misfit1 + 0
            misfit_1st = misfit1_1st + 0
            del popt1; del perr1,; del filename1
        
    print('Final values:')
    print('The following parameters for the covariance function ' + covariance_function(fu)[1] + ' are obtained:')
    if fu in ['gm1', 'gm2', 'reilly', 'hirvonen', 'markov1', 'markov2', 'tri', 'lauer']:
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d'%(popt[0], perr[0], popt[1], perr[1]))
    elif fu == 'vestol':
        print('C0 = %.5f +/- %.5f, a = %.3e +/- %.3e, b = %.3e +/- %.3e'%(popt[0], perr[0], popt[1], perr[1], popt[2], perr[2]))
    elif fu == 'gauss':
        print('C0 = %.5f +/- %.5f, alpha = %.3e +/- %.3e'%(popt[0], perr[0], popt[1], perr[1]))
    elif fu == 'log':
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d, m = %.3f +/- %.3f'%(popt[0], perr[0], popt[1], perr[1], popt[2], perr[2]))
    print('Misfit is: %.5f'%(misfit))
    print('Misfit of the first three points is: %.5f'%(misfit_1st))
    print('Pearsons correlation: %.3f'%(pearsonr(cova, curve)[0]))
    headerline = 'Distance Empirical-Cov Std Number-of-velocities Cov-function'
    np.savetxt('results/' + filename + '_both.dat', np.c_[cova_dist_d, cova, cova_error, cova_length, curve[:]], fmt='%.2f %.4f %.4f %d %.4f', header=headerline)
    print('File %s_both.dat in results/ created'%(filename))
    return popt, perr;
