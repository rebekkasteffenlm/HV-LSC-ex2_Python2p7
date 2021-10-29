#!/usr/bin/python

from scipy.spatial import ConvexHull
import numpy as np

def in_hull(points, point):
    hull = ConvexHull(points)
    new_hull = ConvexHull(np.vstack((points, point)))
    if list(hull.vertices) == list(new_hull.vertices):
        return True
    else:
        return False;
