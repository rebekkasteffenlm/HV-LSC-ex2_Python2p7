#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

def plot_covariance(fs, fl, popt, cova, cova_dist, cova_err, cova_th, filename):
    '''
    Plot covariance function together with data from observations
    '''
    if fs in ['gm1', 'gm2', 'reilly', 'hirvonen', 'markov1', 'markov2', 'tri', 'lauer']:
        C0 = popt[0]
        d0 = popt[1]
        title_plot = 'Covariance function determination using a %s function'%(fl)
        label_plot = r'$\mathregular{C_0}$ = %.3f, $\mathregular{d_0}$ = %d'%(C0, d0)
    elif fs == 'vestol':
        C0 = popt[0]
        a = popt[1]
        b = popt[2]
        title_plot = 'Covariance function determination using a %s function'%(fl)
        label_plot = r'$\mathregular{C_0}$ = %.3f, a = %.3e, b = %.3e'%(C0, a, b)
    elif fs == 'gauss':
        C0 = popt[0]
        alpha = popt[1]
        title_plot = 'Covariance function determination using a %s function'%(fl)
        label_plot = r'$\mathregular{C_0}$ = %.3f, $\alpha$ = %.3e'%(C0, alpha)
    elif fs == 'log':
        C0 = popt[0]
        d0 = popt[1]
        m = popt[2]
        title_plot = 'Covariance function determination using a %s function'%(fl)
        label_plot = '$\mathregular{C_0}$ = %.3f, $\mathregular{d_0}$ = %d, m = %.3f'%(C0, d0, m)
    
    plt.figure(figsize=(8,4))
    plt.errorbar(cova_dist, cova, xerr=0, yerr=cova_err, fmt='+', ecolor='darkgrey', elinewidth=1, capsize=6, capthick=1,
                 marker='+', ms=10, mfc='black', mec='black', label='Estimated covariogram')
    plt.scatter(cova_dist, cova, s=100, c='black', marker='+', lw=2.0, zorder=2)
    plt.plot(cova_dist, cova_th, c='red', lw=2.5, label=label_plot, zorder=3)
    plt.title(title_plot, fontsize=15, y=1.05)
    plt.xlabel('Distance [km]', fontsize=13)
    plt.ylabel(r'Covariance [$\mathregular{\frac{mm^2}{a^2}}$]', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    xmax = np.ceil(np.nanmax(cova_dist) / 10.**(len(str(np.nanmax(cova_dist)).split('.')[0]) - 1)) * 10.**(len(str(np.nanmax(cova_dist)).split('.')[0]) - 1)
    plt.xlim(0 - 10, np.nanmax(cova_dist) + 10)
    plt.legend(loc='upper right')
    plt.savefig('figures/' + str(filename) + '.png', dpi=600, orientation='portrait', format='png', bbox_inches='tight')
    plt.clf()
    print 'Figure %s in figures/ created'%(str(filename) + '.png')
    return;
