#!/usr/bin/python

import argparse
import subprocess
import os
import sys
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from pyproj import Geod
g = Geod(ellps='WGS84')

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

parser.add_argument('-jl', action='store_true', help='Include if covariance matrices should be calculated after JL\n'
                    + '(also add covariance in modus, if the covariance parameters should be calculated after JL)')

parser.add_argument('-mov', nargs='*', help='Include if covariance matrices should be calculated including a moving variance and add factor for delta (after Jonas)\n', metavar='moving-variance')

parser.add_argument('-pb', nargs='*', help='Include if plate boundaries should be included\n'
                    + '(provide list of filename with boundaries, longitude column, latitude column)', metavar='plate_boundaries')

parser.add_argument('-mb', nargs='*', help='Include if different map boundaries should be used for plotting\n'
                    + '(provide list of filename, longitude column, latitude column, number of rows to skip)', metavar='map_boundary')

parser.add_argument('-nc', action='append', nargs='*', type=int, help='Number of the amount of cores to be used in parallelization', metavar='number_cores')

parser.add_argument('-fn', type=str, help='File name of grid points in results/ folder', metavar='file_name')

parser.add_argument('-bo', type=int, default=100000, help='Radius of the circle, which should include stations within', metavar='boundary')

if (len(sys.argv) == 1) or ('--help' in sys.argv) or ('-h' in sys.argv):
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
num_cores = np.array(args.nc[0])
if num_cores.min() == 0:
    num_cores = [multiprocessing.cpu_count(), multiprocessing.cpu_count()]
    num_cores = num_cores[0]

file_name = args.fn
a = file_name.split('/')[:-1]
folder_name = ''
for i in a:
    folder_name = folder_name + i + '/'

bound = args.bo


def jackknife_resampling_res(i, ij_set, jackknife_result_ij, datai, data, l, n, string, bound, num_cores):
    locs = np.where(ij_set[:,0] == i)[0]
    if len(locs) != 0:
        b = jackknife_result_ij[locs]
    else:
        dist = []
        for j in xrange(0, len(data)):
            dist.append(g.inv(data[j,0], data[j,1], datai[i,0], datai[i,1])[2] / 1000.)
        dist = np.array(dist)
        new_bound = np.round(dist[np.argsort(dist)][0], 0) + 100
        b = Parallel(n_jobs=num_cores)(delayed(jackknife_resampling)((j, i), data, [datai[i,0], datai[i,1]], l, n, string) for j in xrange(len(data)) if g.inv(data[j,0], data[j,1], datai[i,0], datai[i,1])[2] / 1000. <= new_bound)
        b = np.array(b)
    a = []
    a.extend([datai[i,0], datai[i,1]])
    for k in xrange(l.shape[1]):
        a.extend([b[:,k].min(), b[:,k].max(), np.mean(b[:,k]), np.std(b[:,k])])
    return a;




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
print('The following file is used now: %s'%(file_inp_info[0]))
file = np.loadtxt('data/' + file_inp_info[0], dtype='S', skiprows=file_inp_info[1])
lon = file[:, file_inp_info[2] - 1].astype(float)
lat = file[:, file_inp_info[3] - 1].astype(float)
if args.pb == None:
    lon_orig = lon + [0]
    lat_orig = lat + [0]
l = np.empty((len(lon), 0))
for i in file_inp_info[4]:
    l = np.c_[l, file[:, i - 1].astype(float)]
if args.err != None:
    n = np.empty((len(lon), 0))
    for i in file_inp_info[5]:
        n = np.c_[n, np.nan_to_num(file[:, i - 1].astype(float))]
else:
    n = [0]

if args.pb != None:
    file_name_pb = file_name.split('.dat')[0] + '_w-pb.dat'
    grid_points = np.loadtxt('results/' + file_name_pb, dtype='S')
    for i in xrange(len(grid_points)):
        grid_points[i,2] = grid_points[i,2] + '\n'
    loni = grid_points[len(lon):,0].astype(float)
    lati = grid_points[len(lon):,1].astype(float)
    grid_points_orig = np.loadtxt('results/' + file_name)
    loni_orig = grid_points_orig[:,0]
    lati_orig = grid_points_orig[:,1]
else:
    grid_points = np.loadtxt('results/' + file_name)
    loni = grid_points[:,0]
    lati = grid_points[:,1]
    loni_orig = loni + [0]
    lati_orig = lati + [0]


string = 'python run_collocation.py -m collocation'
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
    elif vars(args).keys()[i] == 'bo':
        continue
    elif vars(args).keys()[i] == 'fn':
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


ij_set = [(i,j) for i in xrange(len(loni)) for j in xrange(len(lon)) if g.inv(lon[j], lat[j], loni[i], lati[i])[2] / 1000. <= bound]
ij_set = np.array(ij_set)
jackknife_result_ij = Parallel(n_jobs=num_cores)(delayed(jackknife_resampling)((ij[1], ij[0]), np.c_[lon, lat], [loni[ij[0]], lati[ij[0]]], l, n, string) for ij in ij_set)
jackknife_result_ij = np.array(jackknife_result_ij)
jackknife_result = Parallel(n_jobs=num_cores)(delayed(jackknife_resampling_res)(i, ij_set, jackknife_result_ij, np.c_[loni, lati], np.c_[lon, lat], l, n, string, bound, num_cores) for i in xrange(len(loni)))
jackknife_result = np.array(jackknife_result)
jackknife_result[:,0] = loni_orig
jackknife_result[:,1] = lati_orig

formatline = '%.3f %.3f'
headerline = 'lon lat'
for j in xrange(l.shape[1]):
    headerline = headerline + ' min max mean std'
    formatline = formatline + ' %.4f %.4f %.4f %.4f'
np.savetxt('results/' + folder_name + 'jackknife_results_bound%d.dat'%(int(bound)), jackknife_result, fmt=formatline, header=headerline)
