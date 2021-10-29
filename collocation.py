#!/usr/bin/python

import argparse
import sys
import os
from scipy import interpolate
import numpy.ma as ma
from joblib import Parallel, delayed
import multiprocessing
import subprocess

from collocation import *
from covariance import *
from plotting import *
from rest import *
R = 6371000
noise_min = 0.0
new_point_limit = 2000
max_core = 32

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

parser.add_argument('-me', action='store_true', help='Include if mean value should be subtracted before the collocation is done (only mean or median allowed)')

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

parser.add_argument('-p', nargs='*', help='Define the location of the new points:\n'
                    + '- \'auto\': a grid is estimate based on the map boundaries with a chosen grid increment in km\n'
                    + '          (increment in x-direction, increment in y-direction)\n'
                    + '- \'grid\': a grid is estimate based on input parameters (lower-left longitude, lower-left latitude,\n'
                    + '          upper-right longitude, upper-right latitude, longitude increment, latitude increment)\n'
                    + '- \'txt\': a file with locations is loaded (name of file, longitude column, latitude column)\n'
                    + 'Include ''on-land'' if grid points should be on land only, if not nothing is required',
                    metavar='requested_points')

parser.add_argument('-pp', nargs='*', help='Define parameters for plotting (required, if the modus \'plotting\' is chosen):\n'
                    + 'color map (bwr, plasma, ...), boundaries (''y''(es) or ''n''(o)), label for color bar, label for vector (if vector is plotted)', metavar='plotting_parameters')

parser.add_argument('-jl', action='store_true', help='Include if covariance matrices should be calculated after JL\n'
                    + '(also add covariance in modus, if the covariance parameters should be calculated after JL)')

parser.add_argument('-mov', nargs='*', help='Include if covariance matrices should be calculated including a moving variance:\n'
                    + '- add correlation factor, radius, minimum number of stations and fill value', metavar='moving-variance')

parser.add_argument('-pb', nargs='*', help='Include if plate boundaries should be included\n'
                    + '(provide list of filename with boundaries, longitude column, latitude column)', metavar='plate_boundaries')

parser.add_argument('-mb', nargs='*', help='Include if different map boundaries should be used for plotting\n'
                    + '(provide list of filename, longitude column, latitude column, number of rows to skip)', metavar='map_boundary')

parser.add_argument('-stdo', action='store_true', help='Include if correlation function should be estimated')


if (len(sys.argv) == 1) or ('--help' in sys.argv) or ('-h' in sys.argv):
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
string = 'python run_collocation.py'
for i in xrange(0, len(vars(args))):
    if vars(args).values()[i] == None:
        continue
    elif vars(args).values()[i] is False:
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
print(string)

modus = args.m[0]
clabel_order = args.co[0]
if args.err != None:
    file_inp_info = [args.fi, args.skip, args.lon, args.lat, args.obs[0], args.err[0]]
else:
    file_inp_info = [args.fi, args.skip, args.lon, args.lat, args.obs[0]]
distance_conv = args.dc
delta = args.delta
function = args.cf[0]
if len(args.cf) > 1:
    function_parameter = []
    for i in args.cf[1:]:
        function_parameter.append(float(i))
    function_parameter = np.asarray(function_parameter)
    function_parameter = np.reshape(function_parameter, (len(function_parameter) / 2, 2))
    if len(function_parameter) < len(clabel_order):
        b = np.empty((0, 2))
        k = 1
        while k <= len(clabel_order)**2:
            b = np.concatenate([b, function_parameter])
            k += 1
        function_parameter = b + [0]
if args.bg != None:
    bg_model = [args.bg[0], int(args.bg[1]), int(args.bg[2]), args.bg[3:]]
    bg_type_string = '_bg'
else:
    bg_type_string = ''
if args.p != None:
    if args.p[0] == 'auto':
        new_points = [float(args.p[1]), float(args.p[2])]
    elif args.p[0] == 'grid':
        new_points = [[float(args.p[1]), float(args.p[2])], [float(args.p[3]), float(args.p[4])], [float(args.p[5]), float(args.p[6])]]
    elif args.p[0] == 'txt':
        new_points = [args.p[1], int(args.p[2]), int(args.p[3])]
    else:
        args.p = None
if args.pp != None:
    if (args.pp[1] != 'y') and (args.pp[1] != 'n'):
        plot_para = [args.pp[0], 'y', args.pp[1]]
        if len(args.pp) > 2:
            plot_para.append(args.pp[2])
    else:
        plot_para = [args.pp[0], args.pp[1], args.pp[2]]
        if len(args.pp) > 3:
            plot_para.append(args.pp[3])
if args.jl is True:
    covariance_type = 'jl'
    covariance_type_string = '_jl'
    starting_model = [1., 100000.]
else:
    covariance_type = ''
    covariance_type_string = ''
if args.mov != None:
    C0_mov = np.array(args.mov)[0].astype(float)
    delta_mov = list(np.array(args.mov)[1:3].astype(float))
    if len(args.mov) > 3:
        min_num = np.array(args.mov)[3].astype(int)
        if min_num < 2:
            min_num = 2
        if len(args.mov) > 4:
            fill_val = np.array(args.mov)[4].astype(int)
        else:
            fill_val = 9999
    else:
        min_num = 2
        fill_val = 9999
    moving_var = '_mov' + '_'
    for i in delta_mov:
        moving_var = moving_var + str(int(i)) + '-'
    moving_var = moving_var + str(min_num) + '-' + str(fill_val)
else:
    moving_var = ''
if args.pb != None:
    if len(args.pb) == 5:
        pb_model = [args.pb[0], args.pb[1], int(args.pb[2]), int(args.pb[3]), float(args.pb[4])]
    elif len(args.pb) > 5:
        pb_model = [args.pb[0], args.pb[1], int(args.pb[2]), int(args.pb[3]), args.pb[5]]
    else:
        pb_model = [0]
    pb_plot = [args.pb[1], int(args.pb[2]), int(args.pb[3])]
else:
    pb_plot = [0]
if args.mb != None:
    file = np.loadtxt('data/' + args.mb[0], dtype='S', skiprows=int(args.mb[-1]))
    lon_mb = file[:, int(args.mb[1]) - 1].astype(float)
    lat_mb = file[:, int(args.mb[2]) - 1].astype(float)

if (args.pb != None) and (len(args.pb) > 4):
    folder = file_inp_info[0].split('.')[0] + covariance_type_string + bg_type_string + moving_var + '_pb_' + distance_conv + '_' + '+'.join(clabel_order)
else:
    folder = file_inp_info[0].split('.')[0] + covariance_type_string + bg_type_string + moving_var + '_' + distance_conv + '_' + '+'.join(clabel_order)
if args.stdo is True:
    folder = file_inp_info[0].split('.')[0] + covariance_type_string + bg_type_string + moving_var + '_std' + '_' + distance_conv + '_' + '+'.join(clabel_order)
if os.path.exists('figures/' + folder):
    folder = folder
else:
    os.mkdir('figures/' + folder)
if os.path.exists('results/' + folder):
    folder = folder
else:
    os.mkdir('results/' + folder)
folder = folder + '/'



#####################################################################
#                                                                   #
# In the following:                                                 #
# Observational data are loaded (and plotted)                       #
#                                                                   #
#####################################################################
file = np.loadtxt('data/' + file_inp_info[0], dtype='S', skiprows=file_inp_info[1])
lon = file[:, file_inp_info[2] - 1].astype(float)
lat = file[:, file_inp_info[3] - 1].astype(float)
if args.mb == None:
    lon_mb = lon
    lat_mb = lat
if 'plotting' or 'collocation' in modus:
    figure_filename1 = folder + file_inp_info[0].split('.')[0]
    map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
    plot_stations(np.c_[lon, lat], figure_filename1, map)
if distance_conv == 'utm':
    zone_num = utm.from_latlon(np.mean(lat), np.mean(lon))[2]
    x, y = convert_utm(lon, lat, zone_num)
elif distance_conv == 'sphere':
    x, y = lon, lat
else:
    sys.exit('ERROR: Wrong choice of distance calculation provided. Only "sphere" and "utm" are valid arguments.')
l = np.empty((len(lon), 0))
for i in file_inp_info[4]:
    l = np.c_[l, file[:, i - 1].astype(float)]
if args.err != None:
    n = np.empty((len(lon), 0))
    for i in file_inp_info[5]:
        n = np.c_[n, np.nan_to_num(file[:, i - 1].astype(float))]
        n[n <= noise_min] = noise_min
else:
    n = np.zeros(l.shape) + noise_min
print('Input file %s was successfully loaded'%(file_inp_info[0]))
if 'plotting' in modus:
    plot_data(lon, lat, l, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
    plot_data(lon, lat, n, figure_filename1 + '_error', plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
    if l.shape[1] > 1:
        map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
        clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
        plot_vectors_ellipse(np.c_[lon, lat, l[:,clabel_order.index('EW')], l[:,clabel_order.index('NS')], n[:,clabel_order.index('EW')],
                             n[:,clabel_order.index('NS')]], figure_filename1 + clabels, plot_para[3], map)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Check if background model is provided                             #
#                                                                   #
# If background model is provided, the model is loaded and          #
# interpolated for all data to the locations of the observations    #
# --> the observations are corrected for the model                  #
#                                                                   #
#####################################################################
if args.bg != None:
    lonbg, latbg, model, l_corrected, modelxy = apply_backgroundmodel(bg_model, l, distance_conv, x, y)
    
    filename = '%s_obs-loc.dat'%(bg_model[0].split('.')[0])
    formatline = '%.3f %.3f'
    headerline = 'lon lat'
    for i in xrange(l.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_obs'
    for i in xrange(modelxy.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_bg'
    for i in xrange(l_corrected.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_obs-bg'
    np.savetxt('results/' + folder + filename, np.c_[lon, lat, l, modelxy, l_corrected], fmt=formatline, header=headerline)
    if 'plotting' in modus:
        plot_data(lon, lat, modelxy, folder + bg_model[0].split('.')[0] + '_obs-points', plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
    l = l_corrected + 0
    print('Background model %s was successfully loaded and applied to the observations'%(bg_model[0]))
    if 'plotting' in modus:
        figure_filename1 = figure_filename1 + '_bg'
        plot_data(lon, lat, l, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
else:
    print('No background model was included.')



#####################################################################
#                                                                   #
# In the following:                                                 #
# Observational data are reduced by the mean value                  #
#                                                                   #
#####################################################################
if args.me is True:
    mean_l = [np.mean(l[:,i]) for i in xrange(l.shape[1])]
    l = l - mean_l
    print('The observational data were reduced by the following mean values:')
    for i in xrange(len(mean_l)):
        print('%s: %.4f'%(clabel_order[i], mean_l[i]))
    if 'plotting' in modus:
        figure_filename1 = figure_filename1 + '_mean'
        plot_data(lon, lat, l, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Locations for requested points are created or loaded              #
#                                                                   #
#####################################################################
if args.p != None:
    if args.p[0] == 'auto':
        map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1])
        new_points = [[0, 0], [w, h], [new_points[0], new_points[1]]]
        x0 = new_points[0][0]
        y0 = new_points[0][1]
        x1 = new_points[1][0]
        y1 = new_points[1][1]
        dx = new_points[2][0]
        dy = new_points[2][1]
        xs = np.arange(x0, x1 + dx, dx)
        ys = np.arange(y0, y1 + dy, dy)
        xi, yi = np.meshgrid(xs, ys)
        xnew = []; ynew = []
        if args.p[-1] == 'on-land':
            for i in xrange(len(xi.ravel())):
                if map.is_land(xi.ravel()[i] * 1000, yi.ravel()[i] * 1000.) == True:
                    xnew.append(xi.ravel()[i] * 1000.)
                    ynew.append(yi.ravel()[i] * 1000.)
            xi = np.array(xnew)
            yi = np.array(ynew)
        else:
            xi = xi.ravel() * 1000.
            yi = yi.ravel() * 1000.
        loni, lati = map(xi, yi, inverse=True)
        if distance_conv == 'utm':
            xi, yi = convert_utm(loni, lati, zone_num)
        elif distance_conv == 'sphere':
            xi, yi = loni, lati
        print('A grid with the following boundaries was created:')
        print('Lower-left corner: %.2f longitude, %.2f latitude'%(map(0, 0, inverse=True)))
        print('Upper-right corner: %.2f longitude, %.2f latitude'%(map(w * 1000., h * 1000., inverse=True)))
        plt.clf()
        spacing_text = str(new_points[-1][0] * 0.5 + new_points[-1][1] * 0.5).split('.')
        spacing_text = spacing_text[0] + 'p' + spacing_text[1]
        figure_filename2 = folder + file_inp_info[0].split('.')[0] + '_np' + spacing_text
        filenameii = '%s_%s_collocation_np%s'%(file_inp_info[0].split('.')[0], function, spacing_text)
    elif args.p[0] == 'grid':
        map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1])
        lon0 = new_points[0][0]
        lat0 = new_points[0][1]
        lon1 = new_points[1][0]
        lat1 = new_points[1][1]
        dlon = new_points[2][0]
        dlat = new_points[2][1]
        lons = np.arange(lon0, lon1 + dlon, dlon)
        lats = np.arange(lat0, lat1 + dlat, dlat)
        loni, lati = np.meshgrid(lons, lats)
        xnew = []; ynew = []
        if args.p[-1] == 'on-land':
            for i in xrange(len(loni.ravel())):
                a, b = map(loni.ravel()[i], lati.ravel()[i])
                if map.is_land(a, b) == True:
                    xnew.append(loni.ravel()[i])
                    ynew.append(lati.ravel()[i])
            loni = np.array(xnew)
            lati = np.array(ynew)
        else:
            loni = loni.ravel()
            lati = lati.ravel()
        if distance_conv == 'utm':
            xi, yi = convert_utm(loni, lati, zone_num)
        elif distance_conv == 'sphere':
            xi, yi = loni, lati
        print('A grid with the following boundaries was created:')
        print('Lower-left corner: %.2f longitude, %.2f latitude'%(np.amin(loni), np.amin(lati)))
        print('Upper-right corner: %.2f longitude, %.2f latitude'%(np.amax(loni), np.amax(lati)))
        spacing_text = str(new_points[-1][0] * 0.5 + new_points[-1][1] * 0.5).split('.')
        spacing_text = spacing_text[0] + 'p' + spacing_text[1]
        figure_filename2 = folder + file_inp_info[0].split('.')[0] + '_np' + spacing_text
        filenameii = '%s_%s_collocation_np%s'%(file_inp_info[0].split('.')[0], function, spacing_text)
    elif args.p[0] == 'txt':
        filef = np.loadtxt(new_points[0])
        if len(filef.shape) == 1:
            loni = np.array([filef[new_points[1] - 1]])
            lati = np.array([filef[new_points[2] - 1]])
        else:
            loni = filef[:, new_points[1] - 1]
            lati = filef[:, new_points[2] - 1]
        xnew = []; ynew = []
        if args.p[-1] == 'on-land':
            for i in xrange(len(loni)):
                a, b = map(loni[i], lati[i])
                if map.is_land(a, b) == True:
                    xnew.append(loni[i])
                    ynew.append(lati[i])
            loni = np.array(xnew)
            lati = np.array(ynew)
        if distance_conv == 'utm':
            xi, yi = convert_utm(loni, lati, zone_num)
        elif distance_conv == 'sphere':
            xi, yi = loni, lati
        print('The file %s with the requested points was successfully loaded.'%(new_points[0]))
        figure_filename2 = folder + file_inp_info[0].split('.')[0] + '_np-text'
        filenameii = '%s_%s_collocation_np-txt'%(file_inp_info[0].split('.')[0], function)
    else:
        sys.exit('ERROR: Requested points can not be loaded. Only "grid" and "txt" are valid options.')
    map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
    plot_stations(np.c_[loni, lati], figure_filename2, map)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Increase distance for stations on separate plates                 #
#                                                                   #
#####################################################################
if (args.pb != None) and (len(args.pb) > 4):
    plate, boundary_new, map = get_plates(pb_model, lon_mb, lat_mb, plot_para)
    
    if len(args.pb) == 5:
        if args.p == None:
            lon_mod, lat_mod, plate_loc = change_coords_plate(lon, lat, plate, boundary_new, pb_model[4], map)
        elif args.p != None:
            loni_mod, lati_mod, plate_loci = change_coords_plate(lon, lat, plate, boundary_new, pb_model[4], map, loni, lati)
            lon_mod = loni_mod[:len(lon)]
            lat_mod = lati_mod[:len(lat)]
            loni_mod = loni_mod[len(lon):]
            lati_mod = lati_mod[len(lat):]
            plate_loc = plate_loci[:len(lon)]
            plate_loci = plate_loci[len(lon):]
    elif len(args.pb) > 5:
        new_coords = np.loadtxt('results/' + pb_model[4], dtype='S')
        for i in xrange(len(new_coords)):
            new_coords[i,2] = new_coords[i,2] + '\n'
        lon_mod = new_coords[:len(lon),0].astype(float)
        lat_mod = new_coords[:len(lon),1].astype(float)
        plate_loc = new_coords[:len(lon),:]
        if args.p != None:
            loni_mod = new_coords[len(lon):,0].astype(float)
            lati_mod = new_coords[len(lon):,1].astype(float)
            plate_loci = new_coords[len(lon):,:]
    plt.clf()
    filef = open('data/' + file_inp_info[0], 'r')
    a = filef.readlines()
    filef.close()
    skip_text = a[:file_inp_info[1]]
    del a
    filef = open('data/' + file_inp_info[0].split('.dat')[0] + '_w-pb.dat', 'w')
    filef.writelines(skip_text)
    file_new = np.array(file)
    file_new[:, file_inp_info[2] - 1] = lon_mod.astype(str)
    file_new[:, file_inp_info[3] - 1] = lat_mod.astype(str)
    np.savetxt(filef, file_new, fmt='%s')
    filef.close()
    if 'plotting' in modus:
        map, w, h = plotting_area(lon_mod, lat_mod, plot_para[1], pb_plot)
        plot_stations(np.c_[lon_mod, lat_mod], figure_filename1 + '_new', map)
        map, w, h = plotting_area(lon_mod, lat_mod, plot_para[1], pb_plot)
        plot_new_stations(np.c_[lon, lat], np.c_[lon_mod, lat_mod], plate, plate_loc, figure_filename1 + '_old+new', map)
    if distance_conv == 'utm':
        x_mod, y_mod = convert_utm(lon_mod, lat_mod, zone_num)
    elif distance_conv == 'sphere':
        x_mod, y_mod = lon_mod, lat_mod
    if args.p != None:
        np.savetxt('results/' + folder + filenameii + '_w-pb.dat', np.c_[np.hstack((lon_mod, loni_mod)), np.hstack((lat_mod, lati_mod)), np.hstack((plate_loc[:,2], plate_loci[:,2]))], fmt='%s %s %s', newline='')
        if 'plotting' in modus:
            map, w, h = plotting_area(lon_mod, lat_mod, plot_para[1], pb_plot)
            plot_stations(np.c_[loni_mod, lati_mod], figure_filename2 + '_new', map)
            map, w, h = plotting_area(lon_mod, lat_mod, plot_para[1], pb_plot)
            plot_new_stations(np.c_[loni, lati], np.c_[loni_mod, lati_mod], plate, plate_loci, figure_filename2 + '_old+new', map)
            map, w, h = plotting_area(loni_mod, lati_mod, plot_para[1], pb_plot)
            plot_stations(np.c_[loni_mod, lati_mod], figure_filename2 + '_new_all', map)
            map, w, h = plotting_area(loni_mod, lati_mod, plot_para[1], pb_plot)
            plot_new_stations(np.c_[loni, lati], np.c_[loni_mod, lati_mod], plate, plate_loci, figure_filename2 + '_old+new_all', map)
            map, w, h = plotting_area(loni_mod, lati_mod, plot_para[1], pb_plot)
            plot_stations(np.c_[lon_mod, lat_mod], figure_filename1 + '_new_all', map)
            map, w, h = plotting_area(loni_mod, lati_mod, plot_para[1], pb_plot)
            plot_new_stations(np.c_[lon, lat], np.c_[lon_mod, lat_mod], plate, plate_loc, figure_filename1 + '_old+new_all', map)
        if distance_conv == 'utm':
            xi_mod, yi_mod = convert_utm(loni_mod, lati_mod, zone_num)
        elif distance_conv == 'sphere':
            xi_mod, yi_mod = loni_mod, lati_mod
    print('Coordinates have been transformed with respect to plate boundaries according to file %s.'%(pb_model[1]))
else:
    x_mod = x + [0]
    y_mod = y + [0]
    if ('collocation' in modus) and (args.p != None):
        xi_mod = xi + [0]
        yi_mod = yi + [0]



#####################################################################
#                                                                   #
# In the following:                                                 #
# Calculate empirical covariances and find covariance function      #
#                                                                   #
#####################################################################
if ('collocation' in modus) or ('covariance' in modus):
    function_parameters = np.empty((0, 2))
    function_order = []
    if (len(args.cf) > 1) and ((covariance_type != 'jl' and 'covariance' in modus) or ('collocation' in modus)):
        k = 0
        for i, j in itertools.product(xrange(0, l.shape[1]), xrange(0, l.shape[1])):
            if (clabel_order[i] == 'UP' and clabel_order[j] != 'UP') or \
               (clabel_order[i] != 'UP' and clabel_order[j] == 'UP'):
                function_parameters = np.concatenate([function_parameters, np.array([0, 150])[np.newaxis,:]])
            else:
                function_parameters = np.concatenate([function_parameters, function_parameter[k,np.newaxis]])
                k += 1
            function_order.append([clabel_order[i] + '+' + clabel_order[j], i, j])
    elif (covariance_type == 'jl') and ('covariance' in modus) and (len(args.cf) > 0):
        if 'function_parameter' in locals():
            function_parameter = function_parameter[0,:]
        else:
            function_parameter = empirical_covariance(x, y, l[:,0], l[:,0], n[:,0], n[:,0], delta, distance_conv, function,
                                                      folder + file_inp_info[0].split('.')[0], clabel_order[0] + '+' + clabel_order[0])[0]
        while ((function_parameter[1] - starting_model[1]) > 5) or ((function_parameter[1] - starting_model[1]) < -5):
            starting_model = function_parameter
            if (args.mov != None):
                function_parameter = get_covariance_function4_mov(x, y, l, n, delta, delta_mov[0], min_num, function, function_parameter,
                                                                 folder + file_inp_info[0].split('.')[0], 32)[0]
            elif (args.stdo is True):
                function_parameter = get_covariance_function4_std(x, y, l, n, delta, function, function_parameter,
                                                                 folder + file_inp_info[0].split('.')[0], 32)[0]
            else:
                function_parameter = get_covariance_function4(x, y, l, n, delta, function, function_parameter,
                                                              folder + file_inp_info[0].split('.')[0], 32)[0]
        for i, j in itertools.product(xrange(0, l.shape[1]), xrange(0, l.shape[1])):
            function_parameters = np.concatenate([function_parameters, function_parameter[np.newaxis,:]])
            function_order.append([clabel_order[i] + '+' + clabel_order[j], i, j])
    else:
        for i, j in itertools.product(xrange(0, l.shape[1]), xrange(0, l.shape[1])):
            if (clabel_order[i] == 'UP' and clabel_order[j] != 'UP') or \
               (clabel_order[i] != 'UP' and clabel_order[j] == 'UP'):
                function_parameter = np.array([0, 150])
            else:
                if (args.mov != None) and ('covariance' in modus):
                    function_parameter = empirical_covariance_mov(x_mod, y_mod, l[:,i], l[:,j], n[:,i], n[:,j], delta, delta_mov[0], min_num, distance_conv,
                                         function, folder + file_inp_info[0].split('.')[0], clabel_order[i] + '+' + clabel_order[j])[0]
                elif (args.stdo is True) and ('covariance' in modus):
                    function_parameter = empirical_covariance_std(x_mod, y_mod, l[:,i], l[:,j], n[:,i], n[:,j], delta, distance_conv, function,
                                         folder + file_inp_info[0].split('.')[0], clabel_order[i] + '+' + clabel_order[j])[0]
                else:
                    function_parameter = empirical_covariance(x_mod, y_mod, l[:,i], l[:,j], n[:,i], n[:,j], delta, distance_conv, function,
                                         folder + file_inp_info[0].split('.')[0], clabel_order[i] + '+' + clabel_order[j])[0]
            function_parameters = np.concatenate([function_parameters, function_parameter[np.newaxis,:]])
            function_order.append([clabel_order[i] + '+' + clabel_order[j], i, j])
    function_order = np.asarray(function_order)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Create covariance matrices Css and Cnn                            #
#                                                                   #
#####################################################################
if 'collocation' in modus:
    if args.mov != None:
        if abs(l.min()) > abs(l.max()):
            max_l = abs(l.min())
        else:
            max_l = abs(l.max())
        if function_parameter[:,0].max() <= 1:
            max_l = 1.
        function_parameters[:,0] = C0_mov
        Css = create_css(x_mod, y_mod, l, function, function_parameters, function_order, covariance_type, distance_conv)
        
        C0_movvar = np.ones(len(function_order)) * function_parameters[:,0]
        c1_ss, c2_ss = calc_movvar_ss(x_mod, y_mod, l / max_l, function_order, distance_conv, delta_mov[0], min_num, fill_val, C0_movvar)
        movvar = np.empty((0, len(x)))
        for k in xrange(0, len(function_order)):
            movvar = np.concatenate([movvar, np.outer(c1_ss[:,k], c2_ss[:,k])])
        movvar1 = np.empty((0, len(x) * l.shape[1]))
        for i in xrange(0, l.shape[1]):
            movvar2 = np.empty((len(x), 0))
            for j in xrange(0, l.shape[1]):
                movvar2 = np.concatenate([movvar2, movvar[((i * l.shape[1] + j) * len(x)):(((i * l.shape[1] + j) + 1) * len(x)),:]], axis=1)
            movvar1 = np.concatenate([movvar1, movvar2], axis=0)
        movvar_css = movvar1 + [0]
        print('Moving variance for Css created.')
        Css_mov = Css * movvar_css
        
        Cnn_mov = create_cnn(x_mod, y_mod, n / max_l)
        Css = Css_mov + [0]
        Cnn = Cnn_mov + [0]
        del Css_mov; del Cnn_mov
    else:
        Css = create_css(x_mod, y_mod, l, function, function_parameters, function_order, covariance_type, distance_conv)
        Cnn = create_cnn(x_mod, y_mod, n)
    print('All covariance matrices for all input data are created and merged')



#####################################################################
#                                                                   #
# In the following:                                                 #
# Check for positive definiteness of Css matrix                     #
# (eigenvalues are calculated and symmetry is checked)              #
#                                                                   #
#####################################################################
if 'collocation' in modus:
    eigenvalsh = np.linalg.eigvalsh(Css)
    print('Lowest eigenvalue: %8.5e; Symmetry check: %8.5e'%(np.amin(eigenvalsh), np.amax(np.abs(Css - Css.T))))
    epsilon = 1e-10
    if (np.all(eigenvalsh) >= -epsilon) and (np.amin(eigenvalsh) <= 0) and (np.abs(Css - Css.T) <= 1e-2).all():
        print('Css matrix is positive semi-definite')
    elif np.all(eigenvalsh) > 0 and (np.abs(Css - Css.T) <= 1e-2).all():
        print('Css matrix is positive definite')
    else:
        sys.exit('Css matrix is not positive definite and collocation cannot be done')



#####################################################################
#                                                                   #
# In the following:                                                 #
# Collocation --> estimation of the signal s at the observation     #
# points (x, y)                                                     #        
#                                                                   #
#####################################################################
if 'collocation' in modus:
    Czz = Css + Cnn
    Czz_m1 = np.linalg.inv(Czz)
    l_new = np.empty((0, 1))
    for i in xrange(0, l.shape[1]):
        l_new = np.concatenate((l_new, np.reshape(l[:,i], (len(l), 1))), axis=0)
    signal_xy = calc_s(Css, Czz_m1, l_new, A)
    signal_xy_error = np.reshape(calc_ds(Css, Czz_m1, A), signal_xy.shape)
    if args.mov != None:
        signal_xy_error = signal_xy_error * max_l
    noise_xy = calc_n(Cnn, Czz_m1, l_new, A)
    signal_xy_new = np.empty((len(l), 0))
    for i in xrange(0, l.shape[1]):
        signal_xy_new = np.concatenate((signal_xy_new, signal_xy[(i * len(l)):((i + 1) * len(l))]), axis=1)
    signal_xy = signal_xy_new + [0]
    del signal_xy_new
    signal_xy_error_new = np.empty((len(l), 0))
    for i in xrange(0, l.shape[1]):
        signal_xy_error_new = np.concatenate((signal_xy_error_new, signal_xy_error[(i * len(l)):((i + 1) * len(l))]), axis=1)
    signal_xy_error = signal_xy_error_new + [0]
    del signal_xy_error_new
    noise_xy_new = np.empty((len(l), 0))
    for i in xrange(0, l.shape[1]):
        noise_xy_new = np.concatenate((noise_xy_new, noise_xy[(i * len(l)):((i + 1) * len(l))]), axis=1)
    noise_xy = noise_xy_new + [0]
    del noise_xy_new
    print('Collocation at the observation points is done.')
    
    filename = '%s_%s_collocation'%(file_inp_info[0].split('.')[0], function)
    formatline = '%.3f %.3f'
    headerline = 'lon lat'
    for i in xrange(signal_xy.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal'
    for i in xrange(signal_xy_error.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
    for i in xrange(l.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_obs'
    for i in xrange(n.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_err'
    np.savetxt('results/' + folder + filename + '.dat', np.c_[lon, lat, signal_xy, signal_xy_error, l, n], fmt=formatline, header=headerline)
    print('File %s in results/ created.'%(filename + '.dat'))
    if 'plotting' in modus:
        figure_filename1 = folder + file_inp_info[0].split('.')[0] + '_coll'
        plot_data(lon, lat, signal_xy, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        figure_filename1_err = figure_filename1 + '_error'
        plot_data(lon, lat, signal_xy_error, figure_filename1_err, plot_para, clabel_order, pb_plot, lon, lat, n, lon_mb, lat_mb)
        if l.shape[1] > 1:
            map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
            clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
            plot_vectors_ellipse(np.c_[lon, lat, signal_xy[:,clabel_order.index('EW')], signal_xy[:,clabel_order.index('NS')], signal_xy_error[:,clabel_order.index('EW')],
                                 signal_xy_error[:,clabel_order.index('NS')]], figure_filename1 + clabels, plot_para[3], map)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Create covariance matrices Cps and Cpp are created                #
# Collocation of new points is performed, plotted and saved         #
#                                                                   #
#####################################################################
if ('collocation' in modus) and (args.p != None):
    num_loop = int(np.ceil(len(xi_mod) / float(new_point_limit)))
    if num_loop <= max_core:
        loop_cores = num_loop
    else:
        loop_cores = max_core
    if args.mov != None:
        signal_parallel = Parallel(n_jobs=loop_cores)(delayed(calc_signal_xiyi)(x_mod, y_mod, xi_mod, yi_mod, l, Css, Czz_m1, A, function, function_parameters,
                          function_order, covariance_type, distance_conv, new_point_limit, k, [delta_mov, min_num, fill_val, C0_movvar, c2_ss], function_parameter) for k in xrange(num_loop))
    else:
        signal_parallel = Parallel(n_jobs=loop_cores)(delayed(calc_signal_xiyi)(x_mod, y_mod, xi_mod, yi_mod, l, Css, Czz_m1, A, function, function_parameters,
                          function_order, covariance_type, distance_conv, new_point_limit, k) for k in xrange(num_loop))
    signal_xiyi = np.empty((0, l.shape[1])); signal_xiyi_error = np.empty((0, n.shape[1]))
    for k in xrange(num_loop):
        signal_xiyi = np.concatenate((signal_xiyi, signal_parallel[k][0]), axis=0)
        signal_xiyi_error = np.concatenate((signal_xiyi_error, signal_parallel[k][1]), axis=0)
    print('Collocation at new points is done.')
    
    formatline = '%.3f %.3f'
    headerline = 'lon lat'
    for i in xrange(signal_xiyi.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal'
    for i in xrange(signal_xiyi_error.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
    np.savetxt('results/' + folder + filenameii + '.dat', np.c_[loni, lati, signal_xiyi, signal_xiyi_error], fmt=formatline, header=headerline)
    print('File %s in results/ created'%(filenameii + '.dat'))
    if 'plotting' in modus:
        figure_filename2 = figure_filename2 + '_coll'
        plot_data(loni, lati, signal_xiyi, figure_filename2, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        figure_filename2_err = figure_filename2 + '_error'
        plot_data(loni, lati, signal_xiyi_error, figure_filename2_err, plot_para, clabel_order, pb_plot, lon, lat, n, lon_mb, lat_mb)



#####################################################################
#                                                                   #
# In the following:                                                 #
# Mean value is added to new data (s and sp)                        #
#                                                                   #
#####################################################################
if (args.me is True) and ('collocation' in modus):
    l = l + mean_l
    signal_xy = signal_xy + mean_l
    if 'plotting' in modus:
        figure_filename1 = figure_filename1 + '_mean'
        plot_data(lon, lat, signal_xy, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        if l.shape[1] > 1:
            map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
            clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
            plot_vectors_ellipse(np.c_[lon, lat, signal_xy[:,clabel_order.index('EW')], signal_xy[:,clabel_order.index('NS')], signal_xy_error[:,clabel_order.index('EW')],
                                 signal_xy_error[:,clabel_order.index('NS')]], figure_filename1 + clabels, plot_para[3], map)
    print('Mean value is added to new signal values and results are plotted.')
    filename = filename + '_mean'
    formatline = '%.3f %.3f'
    headerline = 'lon lat'
    for i in xrange(signal_xy.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal'
    for i in xrange(signal_xy_error.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
    for i in xrange(l.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_obs'
    for i in xrange(n.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_err'
    np.savetxt('results/' + folder + filename + '.dat', np.c_[lon, lat, signal_xy, signal_xy_error, l, n], fmt=formatline, header=headerline)
    print('File %s in results/ created'%(filename + '.dat'))
    
    if args.p != None:
        signal_xiyi = signal_xiyi + mean_l
        if 'plotting' in modus:
            figure_filename2 = figure_filename2 + '_mean'
            plot_data(loni, lati, signal_xiyi, figure_filename2, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        print('Mean value is added to new signal values and results are plotted.')
        filenameii = filenameii + '_mean'
        formatline = '%.3f %.3f'
        headerline = 'lon lat'
        for i in xrange(signal_xiyi.shape[1]):
            formatline = formatline + ' %.4f'
            headerline = headerline + ' ' + clabel_order[i] + '_signal'
        for i in xrange(signal_xiyi_error.shape[1]):
            formatline = formatline + ' %.4f'
            headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
        np.savetxt('results/' + folder + filenameii + '.dat', np.c_[loni, lati, signal_xiyi, signal_xiyi_error], fmt=formatline, header=headerline)
        print('File %s in results/ created'%(filenameii + '.dat'))



#####################################################################
#                                                                   #
# In the following:                                                 #
# Check if background model is provided                             #
#                                                                   #
# If background model is provided, the obtained signals are         #
# corrected for the model                                           #
#                                                                   #
#####################################################################
if ('collocation' in modus) and (args.bg != None):
    signal_xy_corrected = signal_xy + 0
    l_corrected = l + 0
    if distance_conv == 'utm':
        xn = x + 0
        yn = y + 0
    elif distance_conv == 'sphere':
        zone_num = utm.from_latlon(np.mean(x), np.mean(y))[2]
        xn, yn = convert_utm(x, y, zone_num)
    xm, ym = convert_utm(lonbg, latbg, zone_num)
    modelxy = np.empty((len(lon), 0))
    for i in xrange(0, model.shape[1]):
        model_xy = interpolate.griddata((xm, ym), model[:,i], (xn, yn), method='cubic')
        for j in xrange(len(model_xy)):
            if in_hull(np.c_[xm, ym], np.c_[xn[j], yn[j]]) is False:
                model_xy[j] = 0.
        if len(np.argwhere(np.isnan(model_xy))) > 0:
            fspline = interpolate.SmoothBivariateSpline(xn[np.isfinite(model_xy)], yn[np.isfinite(model_xy)],
                                                        model_xy[np.isfinite(model_xy)], kx=5, ky=5)
            for j in np.argwhere(np.isnan(model_xy)):
                model_xy[j] = fspline(xn[j], yn[j])[0][0]
        signal_xy_corrected[:,i] = signal_xy_corrected[:,i] + model_xy[:]
        l_corrected[:,i] = l_corrected[:,i] + model_xy[:]
    signal_xy = signal_xy_corrected + 0
    l = l_corrected + 0
    print('Background model %s was applied to the obtained signals at the observation points'%(bg_model[0]))
    if 'plotting' in modus:
        figure_filename1 = figure_filename1 + '_bg'
        plot_data(lon, lat, signal_xy, figure_filename1, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        if l.shape[1] > 1:
            map, w, h = plotting_area(lon_mb, lat_mb, plot_para[1], pb_plot)
            clabels = '_' + clabel_order[clabel_order.index('EW')] + '+' + clabel_order[clabel_order.index('NS')]
            plot_vectors_ellipse(np.c_[lon, lat, signal_xy[:,clabel_order.index('EW')], signal_xy[:,clabel_order.index('NS')], signal_xy_error[:,clabel_order.index('EW')],
                                 signal_xy_error[:,clabel_order.index('NS')]], figure_filename1 + clabels, plot_para[3], map)
    filename = filename + '_bg'
    formatline = '%.3f %.3f'
    headerline = 'lon lat'
    for i in xrange(signal_xy.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal'
    for i in xrange(signal_xy_error.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
    for i in xrange(l.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_obs'
    for i in xrange(n.shape[1]):
        formatline = formatline + ' %.4f'
        headerline = headerline + ' ' + clabel_order[i] + '_err'
    np.savetxt('results/' + folder + filename + '.dat', np.c_[lon, lat, signal_xy, signal_xy_error, l, n], fmt=formatline, header=headerline)
    print('File %s in results/ created'%(filename + '.dat'))
    
    if args.p != None:
        signal_xiyi_corrected = signal_xiyi + 0
        if distance_conv == 'utm':
            xn = xi + 0
            yn = yi + 0
        elif distance_conv == 'sphere':
            zone_num = utm.from_latlon(np.mean(xi), np.mean(yi))[2]
            xn, yn = convert_utm(xi, yi, zone_num)
        xm, ym = convert_utm(lonbg, latbg, zone_num)
        modelxy = np.empty((len(lon), 0))
        for i in xrange(0, model.shape[1]):
            model_xy = interpolate.griddata((xm, ym), model[:,i], (xn, yn), method='cubic')
            for j in xrange(len(model_xy)):
                if in_hull(np.c_[xm, ym], np.c_[xn[j], yn[j]]) is False:
                    model_xy[j] = 0.
            if len(np.argwhere(np.isnan(model_xy))) > 0:
                fspline = interpolate.SmoothBivariateSpline(xn[np.isfinite(model_xy)], yn[np.isfinite(model_xy)],
                                                            model_xy[np.isfinite(model_xy)], kx=5, ky=5)
                for j in np.argwhere(np.isnan(model_xy)):
                    model_xy[j] = fspline(xn[j], yn[j])[0][0]
            signal_xiyi_corrected[:,i] = signal_xiyi_corrected[:,i] + model_xy[:]
        signal_xiyi = signal_xiyi_corrected + 0
        print('Background model %s was applied to the obtained signals at the new points'%(bg_model[0]))
        if 'plotting' in modus:
            figure_filename2 = figure_filename2 + '_bg'
            plot_data(loni, lati, signal_xiyi, figure_filename2, plot_para, clabel_order, pb_plot, lon, lat, l, lon_mb, lat_mb)
        filenameii = filenameii + '_bg'
        formatline = '%.3f %.3f'
        headerline = 'lon lat'
        for i in xrange(signal_xiyi.shape[1]):
            formatline = formatline + ' %.4f'
            headerline = headerline + ' ' + clabel_order[i] + '_signal'
        for i in xrange(signal_xiyi_error.shape[1]):
            formatline = formatline + ' %.4f'
            headerline = headerline + ' ' + clabel_order[i] + '_signal-err'
        np.savetxt('results/' + folder + filenameii + '.dat', np.c_[loni, lati, signal_xiyi, signal_xiyi_error], fmt=formatline, header=headerline)
        print('File %s in results/ created'%(filenameii + '.dat'))



#####################################################################
#                                                                   #
# In the following:                                                 #
# Rename folders wrt the covariance function and their parameters   #
#                                                                   #
#####################################################################
if ('collocation' in modus) or ('covariance' in modus):
    if len(args.cf) > 1:
        new_folder = folder.split('/')[0]
        for i in args.cf:
            new_folder = new_folder + '_' + i
    elif (covariance_type == 'jl') and ('covariance' in modus):
        new_folder = folder.split('/')[0] + '_' + args.cf[0] + '_d' + str(delta) + '_covariance-analysis'
    else:
        new_folder = folder.split('/')[0] + '_' + args.cf[0] + '_d' + str(delta)
        new_folder = '%s_C0-%.3f'%(new_folder, float(np.mean(function_parameters[:,0])))
        new_folder = '%s_d0-%d'%(new_folder, float(np.mean(function_parameters[:,1])))
    if new_folder in os.listdir('figures/'):
        print 'Folder already exists! Data are moved to existing folder %s'%(new_folder)
        os.system('mv figures/' + folder + '* figures/' + new_folder + '/.')
        os.system('rm -r figures/' + folder)
        os.system('mv results/' + folder + '* results/' + new_folder + '/.')
        os.system('rm -r results/' + folder)
    else:
        os.rename('figures/' + folder, 'figures/' + new_folder)
        os.rename('results/' + folder, 'results/' + new_folder)
        print('Results and figures are saved in %s'%(new_folder))
