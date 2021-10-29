#!/usr/bin/python

import numpy as np
import scipy

def get_covariance_parameter(cova, cova_dist, corr_bounds, f, fs, fl, filei):
    '''
    Fit function to obtained empirical covariance and covariance distances to find signal covariance
    and correlation length
    '''
    b1 = cova[0]
    print('The following parameters for the covariance function ' + fl + ' are obtained: ')
    if fs in ['gm1', 'gm2', 'reilly', 'hirvonen', 'markov1', 'markov2', 'tri', 'lauer']:
        b3 = corr_bounds[0]
        b4 = corr_bounds[1]
        while 'popt' not in locals():
            try:
                popt, pcov = scipy.optimize.curve_fit(lambda cova_dist, d0: f(cova_dist, b1, d0), cova_dist, cova, bounds=([b3], [b4]))
            except RuntimeError:
                print('Error - curve_fit failed')
                b4 = b4 / 2.
        perr = np.sqrt(np.diag(pcov))
        d0 = popt[0]
        sd0 = perr[0]
        C0 = b1
        sC0 = 0
        popt = np.array([C0, d0])
        perr = np.array([sC0, sd0])
        filename = '%s_cov_%s_C0-%.3f_d0-%d'%(filei, fs, C0, d0)
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d'%(C0, sC0, d0, sd0))
        if (sC0 == np.inf) or (sd0 == np.inf):
            sys.exit('ERROR: The function ' + fl + ' is not an applicable covariance function for the used dataset.')
    elif fs == 'vestol':
        popt, pcov = scipy.optimize.curve_fit(lambda cova_dist, a, b: f(cova_dist, b1, a, b), cova_dist, cova, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
        perr = np.sqrt(np.diag(pcov))
        a = popt[0]
        sa = perr[0]
        b = popt[1]
        sb = perr[1]
        C0 = b1
        sC0 = 0
        popt = np.array([C0, a, b])
        perr = np.array([sC0, sa, sb])
        filename = '%s_cov_%s_C0-%.3f_a-%.3e_b-%.3e'%(filei, fs, C0, a, b)
        print('C0 = %.5f +/- %.5f, a = %.3e +/- %.3e, b = %.3e +/- %.3e'%(C0, sC0, a, sa, b, sb))
        if (sC0 == np.inf) or (sa == np.inf) or (sb == np.inf):
            sys.exit('ERROR: The function ' + fl + ' is not an applicable covariance function for the used dataset.')
    elif fs == 'gauss':
        b3 = 1 / float(corr_bounds[1])
        b4 = 1 / float(corr_bounds[0])
        popt, pcov = scipy.optimize.curve_fit(lambda cova_dist, alpha: f(cova_dist, b1, alpha), cova_dist, cova, bounds=([b3], [b4]))
        perr = np.sqrt(np.diag(pcov))
        alpha = popt[0]
        salpha = perr[0]
        C0 = b1
        sC0 = 0
        popt = np.array([C0, alpha])
        perr = np.array([sC0, salpha])
        filename = '%s_cov_%s_C0-%.3f_alpha-%.3e'%(filei, fs, C0, alpha)
        print('C0 = %.5f +/- %.5f, alpha = %.3e +/- %.3e'%(C0, sC0, alpha, salpha))
        if (sC0 == np.inf) or (salpha == np.inf):
            sys.exit('ERROR: The function ' + fl + ' is not an applicable covariance function for the used dataset.')
    elif fs == 'log':
        b3 = corr_bounds[0]
        b4 = corr_bounds[1]
        popt, pcov = scipy.optimize.curve_fit(lambda cova_dist, d0, m: f(cova_dist, b1, d0, m), cova_dist, cova, bounds=([b3, 0], [b4, 10]))
        perr = np.sqrt(np.diag(pcov))
        d0 = popt[0]
        sd0 = perr[0]
        m = popt[1]
        sm = perr[1]
        C0 = b1
        sC0 = 0
        popt = np.array([C0, d0, m])
        perr = np.array([sC0, sd0, sm])
        filename = '%s_cov_%s_C0-%.3f_d0-%d_m-%.3f'%(filei, fs, C0, d0, m)
        print('C0 = %.5f +/- %.5f, d0 = %d +/- %d, m = %.3f +/- %.3f'%(C0, sC0, d0, sd0, m, sm))
        if (sC0 == np.inf) or (sd0 == np.inf) or (sm == np.inf):
            sys.exit('ERROR: The function ' + fl + ' is not an applicable covariance function for the used dataset.')
    return popt, perr, filename;
