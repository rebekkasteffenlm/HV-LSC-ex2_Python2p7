# This file shows how the Python scripts are started based on the example of a synthetic dataset

# Standard covariance analysis of the horizontal components:
python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m covariance -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 1.5 -cf gm1 -noise 0.3



# Correlation analysis of the horizontal components:
python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m covariance -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 1.5 -cf gm1 -stdo -noise 0.3



# Combined covariance and correlation analysis of the horizontal components:
python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m covariance -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 1.5 -cf gm1 1.969 500 -jl -noise 0.3

python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m covariance -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 1.5 -cf gm1 1. 500 -jl -stdo -noise 0.3



# Collocation of the horizontal velocity field:
python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation plotting -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.025 268 0.0 250 0.0 250 1.913 500 -p 'grid' 10 -25 80 41 0.25 0.25 -noise 0.3

python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation plotting -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.120 250 -jl -p 'grid' 10 -25 80 41 0.25 0.25 -noise 0.3

python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation plotting -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 1500 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.120 250 -jl -p 'grid' 10 -25 80 41 0.25 0.25 -noise 0.3

python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation plotting -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 1500 syn+noise-4plates_rot_10-80_-25-40_130_jl_pb_sphere_EW+NS_gm1_2.120_250/syn+noise-4plates_rot_10-80_-25-40_130_gm1_collocation_np0p25_w-pb.dat -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.120 369 -jl -p 'grid' 10 -25 80 41 0.25 0.25 -mov 1.00 850 850 7 -noise 0.3



# LOOCV
python run_cross-validation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation -lon 1 -dc sphere -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.025 268 0.0 250 0.0 250 1.913 500 -nc 32 -noise 0.3



# Jackknife resampling
python run_collocation.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation plotting -lon 1 -dc sphere -pb 4plates_10-80_-25-40_plates.dat 4plates_10-80_-25-40_boundaries.dat 1 2 -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.025 268 0.0 250 0.0 250 1.913 500 -p 'grid' 10 -25 80 41 2.0 2.0 -noise 0.3
python run_jackknife-resampling.py -me -pp bwr n 'Velocity [mm/a]' mm/a -co EW NS -err 5 6 -m collocation -lon 1 -dc sphere -skip 0 -lat 2 -fi syn+noise-4plates_rot_10-80_-25-40_130.dat -obs 3 4 -delta 0.5 -cf gm1 2.025 268 0.0 250 0.0 250 1.913 500 -nc 40 -fn syn+noise-4plates_rot_10-80_-25-40_130_sphere_EW+NS_gm1_2.025_268_0.0_250_0.0_250_1.913_500/syn+noise-4plates_rot_10-80_-25-40_130_gm1_collocation_np2p0.dat -bo 1500 -noise 0.3
