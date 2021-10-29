#!/usr/bin/python

import numpy as np

def get_cnn(n1, n2):
    '''
    Estimate the Cnn matrix for the collocation (covariance function of the noise)
    '''
    Cnn = np.identity(len(n1)) * (n1 * n2)
    print 'Noise-covariance matrix Cnn created'
    return Cnn;
