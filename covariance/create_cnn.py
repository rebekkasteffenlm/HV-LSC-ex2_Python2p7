#!/usr/bin/python

import numpy as np
import itertools
from get_cnn import *


def create_cnn(x, y, n):
    Cnn = np.empty((0, len(x)))
    for i, j in itertools.product(xrange(0, n.shape[1]), xrange(0, n.shape[1])):
        if (i == j):
            Cnn_p = get_cnn(n[:,i], n[:,j])
        else:
            Cnn_p = np.zeros((len(n), len(n)))
        Cnn = np.concatenate([Cnn, Cnn_p])
        del Cnn_p
    Cnn1 = np.empty((0, len(x) * n.shape[1]))
    for i in xrange(0, n.shape[1]):
        Cnn2 = np.empty((len(x), 0))
        for j in xrange(0, n.shape[1]):
            Cnn2 = np.concatenate([Cnn2, Cnn[((i * n.shape[1] + j) * len(x)):(((i * n.shape[1] + j) + 1) * len(x)),:]], axis=1)
        Cnn1 = np.concatenate([Cnn1, Cnn2], axis=0)
    Cnn = Cnn1 + [0]
    return Cnn;

