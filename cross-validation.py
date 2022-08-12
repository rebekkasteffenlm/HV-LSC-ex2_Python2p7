#!/usr/bin/python

import argparse
import subprocess
import os
import sys
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from rest import *

#####################################################################
#                                                                   #
# In the following:                                                 #
# Read input values                                                 #
#                                                                   #
#####################################################################

parser = argparse.ArgumentParser(description='Collocation script, including determining the parameters of a covariance'
            + ' function and plotting input as well as output data', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-m', type=str, action='append', nargs='*', choices=['collocation', 'plotting', 'covariance'], required=True,
                    help='Purpose of the script (three choices, required):\n'
                    + '- \'plotting\': Input data are plotted\n'
                    + '- \'covariance\': Parameters of a specific covariance function are determined\n'
                    + '- \'collocation\': Collocation is performed (including covariance and plotting)',
                    metavar='script_modus')

parser.add_argument('-fi', type=str, required=True, help='Name of input file (required)', metavar='input_file')
parser.add_argument('-skip', type=int, default=0, help='Number of header lines in input file (default is 0)', metavar='header_lines')
parser.add_argument('-lon', default=1, type=int, help='Column of longitude values (default is 1)', metavar='longitude')
parser.add_argument('-lat', default=2, type=int, help='Column of latitude values (default is 2)', metavar='latitude')
parser.add_argument('-obs', action='append', nargs='*', type=int, required=True, help='Column(s) of observational data (at least one required)',
                    metavar='observations')
parser.add_argument('-err', action='append', nargs='*', type=int, help='Column(s) of data uncertainty', metavar='noise')
parser.add_argument('-co', action='append', nargs='*', type=str, help='Name of components in order provided above (at least one required)',
                    metavar='label')

parser.add_argument('-me', action='store_true', help='Include if mean value should be subtracted before the collocation is done')

parser.add_argument('-dc', type=str, choices=['utm', 'sphere'], default='utm', metavar='distance_calculation',
                    help='Choose the method for calculating distances between two locations (default is utm):\n'
                    + '- \'utm\' (geographic coordinates are converted into UTM coordinates)\n'
                    + '- \'sphere\' (distances are calculated on a sphere using geographic coordinates)')

parser.add_argument('-delta', type=float, help='Define the distance of the intervals used to estimate the covariances (in km or degree)',
                    metavar='delta_covariance')

parser.add_argument('-bg', nargs='*', help='Define the parameters of a background model (if required):\n'
                    + 'Name of background model file, column of longitude values, column of latitude values, column of data\n'
                    + '(the data columns must be in the same order as for the input file)',
                    metavar='background_model')

parser.add_argument('-cf', type=str, nargs='*', default=['gm1'], help='Define covariance function:\n'
                    + '- \'gm1\': Gauss-Markov 1st order (C0 * exp(-d/d0))\n'
                    + '- \'gm2\': Gauss-Markov 2nd order (C0 * exp(-(d/d0)^2))\n'
                    + '- \'reilly\': Reilly (C0 * (1 - 0.5*(d/d0)^2 * exp(-0.5 * (d/d0)^2)))\n'
                    + '- \'hirvonen\': Hirvonen (C0 / (1 + (d/d0)^2))\n'
                    + '- \'markov1\': 1st order Markov (C0 * (1 + d/d0) * exp(-d/d0))\n'
                    + '- \'markov2\': 2nd order Markov (C0 * (1 + d/d0 + (d^2/3d0^2) * exp(-d/d0))\n'
                    + '- \'tri\': Triangular (C0 * (1 - d/2d0))\n'
                    + '- \'lauer\': Lauer (C0 / (d * d0))\n'
                    + '- \'vestol\': (C0 * (a*d^2 + b*d + 1))\n'
                    + '- \'gauss\': C0 * exp(-(alpha * d)^2)\n'
                    + '- \'log\': Logarithmic (C0 / (1 + (d/d0)^m))\n'
                    + 'additionally: covariance function parameters can be provided',
                    metavar='covariance_function')

parser.add_argument('-A', type=str, choices=['polynom', 'ang-vel', 'rotation', 'covariance'],
                    help='Define the values for the A matrix (trend function):\n'
                    + '- \'polynom\': the A matrix is created using high-order polynomials ()\n'
                    + '- \'ang-vel\': the A matrix is created by transforming the velocity to angular velocities (Euler vector)\n'
                    + '- \'rotation\': the A matrix is created by applying a rotation based on barycentric coordinates\n'
                    + '- \'covariance\': the A matrix is created by using a covariance function\n'
                    + 'if -A is omitted, the A matrix will be zero (no trend will be removed from the signal)',
                    metavar='A-matrix')

parser.add_argument('-pp', nargs='*', help='Define parameters for plotting (required, if the modus \'plotting\' is chosen):\n'
                    + 'color map (bwr, plasma, ...), boundaries (''y''(es) or ''n''(o)), label for color bar, label for vector (if vector is plotted)', metavar='plotting_parameters')

parser.add_argument('-jl', action='store_true', help='Include if covariance matrices should be calculated after Legrand (2007)\n'
                    + '(also add covariance in modus, if the covariance parameters should be calculated after Legrand (2007))')

parser.add_argument('-mov', nargs='*', help='Include if covariance matrices should be calculated including a moving variance and add factor for delta (after Jonas)\n', metavar='moving-variance')

parser.add_argument('-pb', nargs='*', help='Include if plate boundaries should be included\n'
                    + '(provide list of filename with boundaries, longitude column, latitude column)', metavar='plate_boundaries')

parser.add_argument('-mb', nargs='*', help='Include if different map boundaries should be used for plotting\n'
                    + '(provide list of filename, longitude column, latitude column, number of rows to skip)', metavar='map_boundary')

parser.add_argument('-noise', default=0.3, type=float, help='Minimum noise level', metavar='noise_minimum')

parser.add_argument('-nc', type=int, default=0, help='Number of the amount of cores to be used in parallelization', metavar='number_cores')


if (len(sys.argv) == 1) or ('--help' in sys.argv) or ('-h' in sys.argv):
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
num_cores = args.nc
if num_cores == 0:
    num_cores = multiprocessing.cpu_count()


#####################################################################
#                                                                   #
# In the following:                                                 #
# Create python command from input command                          #
#                                                                   #
#####################################################################    
string = 'python collocation.py'
for i in xrange(0, len(vars(args))):
    if vars(args).values()[i] == None:
        continue
    elif vars(args).values()[i] is False:
        continue
    elif vars(args).keys()[i] == 'nc':
        continue
    elif vars(args).values()[i] is True:
        string = string + ' -%s'%(vars(args).keys()[i])
    else:
        if isinstance(vars(args).values()[i], list) is True:
            if isinstance(vars(args).values()[i][0], list) is True:
                string = string + ' -%s'%(str(vars(args).keys()[i]))
                for j in vars(args).values()[i][0]:
                    if len(str(j).split()) == 1:
                        string = string + ' ' + str(j)
                    else:
                        string = string + ' "' + ' '.join(j) + '"'
            else:
                string = string + ' -%s'%(str(vars(args).keys()[i]))
                for j in vars(args).values()[i]:
                    if len(str(j).split()) == 1:
                        string = string + ' ' + str(j)
                    else:
                        string = string + ' "' + ' '.join(str(j).split()) + '"'                
        else:
            string = string + ' -%s %s'%(str(vars(args).keys()[i]), str(vars(args).values()[i]))
print string



#####################################################################
#                                                                   #
# In the following:                                                 #
# Run standard collocation and read results                         #
#                                                                   #
#####################################################################            
subprocess.check_call(string + ' > cross-validation_out.dat', shell=True)
print 'Collocation of the main dataset is done'
file = open('cross-validation_out.dat', 'r')
a = file.readlines()
file.close()
folder = a[-1].split()[-1]
files = os.listdir('results/' + folder + '/')
for j in files:
    if '_collocation_mean_bg.dat' in j:
        res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
        l_coll = np.empty((len(res), 0))
        for i in xrange(0, len(args.obs[0])):
            l_coll = np.c_[l_coll, res[:,i+2]]
        break
if 'l_coll' not in locals():
    for j in files:
        if '_collocation_mean.dat' in j:
            res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
            l_coll = np.empty((len(res), 0))
            for i in xrange(0, len(args.obs[0])):
                l_coll = np.c_[l_coll, res[:,i+2]]
            break
if 'l_coll' not in locals():
    for j in files:
        if '_collocation.dat' in j:
            res = np.loadtxt('results/' + folder + '/' + j, skiprows=1)
            l_coll = np.empty((len(res), 0))
            for i in xrange(0, len(args.obs[0])):
                l_coll = np.c_[l_coll, res[:,i+2]]
            break



#####################################################################
#                                                                   #
# In the following:                                                 #
# Read input data and reduce by one station                         #
#                                                                   #
#####################################################################    
if args.err != None:
    file_inp_info = [args.fi, args.skip, args.lon, args.lat, args.obs[0], args.err[0]]
else:
    file_inp_info = [args.fi, args.skip, args.lon, args.lat, args.obs[0]]
if args.pb != None:
    file = np.loadtxt('data/' + file_inp_info[0], dtype='S', skiprows=file_inp_info[1])
    lon_orig = file[:, file_inp_info[2] - 1].astype(float)
    lat_orig = file[:, file_inp_info[3] - 1].astype(float)
    file_inp_info[0] = file_inp_info[0].split('.dat')[0] + '_w-pb.dat'
print 'The following file is used now: %s'%(file_inp_info[0])
file = np.loadtxt('data/' + file_inp_info[0], dtype='S', skiprows=file_inp_info[1])
lon = file[:, file_inp_info[2] - 1].astype(float)
lat = file[:, file_inp_info[3] - 1].astype(float)
l = np.empty((len(lon), 0))
for i in file_inp_info[4]:
    l = np.c_[l, file[:, i - 1].astype(float)]
if args.err != None:
    n = np.empty((len(lon), 0))
    for i in file_inp_info[5]:
        n = np.c_[n, np.nan_to_num(file[:, i - 1].astype(float))]
else:
    n = [0]

string = 'python collocation.py -m collocation'
for i in xrange(0, len(vars(args))):
    if vars(args).values()[i] == None:
        continue
    elif vars(args).values()[i] is False:
        continue
    elif vars(args).values()[i] is True:
        string = string + ' -%s'%(vars(args).keys()[i])
    elif vars(args).keys()[i] == 'nc':
        continue
    elif vars(args).keys()[i] == 'fi':
        continue
    elif vars(args).keys()[i] == 'lon':
        continue
    elif vars(args).keys()[i] == 'lat':
        continue
    elif vars(args).keys()[i] == 'skip':
        continue
    elif vars(args).keys()[i] == 'obs':
        continue
    elif vars(args).keys()[i] == 'err':
        continue
    elif vars(args).keys()[i] == 'pb':
        continue
    elif vars(args).keys()[i] == 'm':
        continue
    elif vars(args).keys()[i] == 'noise':
        continue
    else:
        if isinstance(vars(args).values()[i], list) is True:
            if isinstance(vars(args).values()[i][0], list) is True:
                string = string + ' -%s'%(str(vars(args).keys()[i]))
                for j in vars(args).values()[i][0]:
                    if len(str(j).split()) == 1:
                        string = string + ' ' + str(j)
                    else:
                        string = string + ' "' + ' '.join(j) + '"'
            else:
                string = string + ' -%s'%(str(vars(args).keys()[i]))
                for j in vars(args).values()[i]:
                    if len(str(j).split()) == 1:
                        string = string + ' ' + str(j)
                    else:
                        string = string + ' "' + ' '.join(str(j).split()) + '"'                
        else:
            string = string + ' -%s %s'%(str(vars(args).keys()[i]), str(vars(args).values()[i]))


cross_result = Parallel(n_jobs=num_cores)(delayed(cross_validation)(i, np.c_[lon, lat], l, n, l_coll, string) for i in xrange(0, len(lon)))
cross_result = np.asarray(cross_result)
if args.pb != None:
    cross_result[:,1] = lon_orig
    cross_result[:,2] = lat_orig
formatline = '%d %.3f %.3f'
headerline = 'num lon lat observation collocation interpolation difference_obs-int difference_coll-int'
for j in xrange(l.shape[1]*5):
    formatline = formatline + ' %.4f'
np.savetxt('results/' + folder + '/cross-validation_results.dat', cross_result, fmt=formatline, header=headerline)

print('The following statistical values have been determined for the cross validation (to observations):')
for j in xrange(l.shape[1]):
    maxc = np.amax(cross_result[:,l.shape[1]*4+j+1])
    minc = np.amin(cross_result[:,l.shape[1]*4+j+1])
    mean = np.mean(cross_result[:,l.shape[1]*4+j+1])
    std = np.std(cross_result[:,l.shape[1]*4+j+1])
    print('Component: %s; Maximum: %.4f; Minimum: %.4f; Mean: %.4f; Std: %.4f'%(args.co[0][j], maxc, minc, mean, std))

print('The following statistical values have been determined for the cross validation (to signals):')
for j in xrange(l.shape[1]):
    maxc = np.amax(cross_result[:,l.shape[1]*5+j+1])
    minc = np.amin(cross_result[:,l.shape[1]*5+j+1])
    mean = np.mean(cross_result[:,l.shape[1]*5+j+1])
    std = np.std(cross_result[:,l.shape[1]*5+j+1])
    print('Component: %s; Maximum: %.4f; Minimum: %.4f; Mean: %.4f; Std: %.4f'%(args.co[0][j], maxc, minc, mean, std))

print('------------------------------------------------------------------------------')
subprocess.check_call('rm cross-validation_out.dat', shell=True)
