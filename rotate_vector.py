# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:33:18 2021

@author: OBRIEJ25
"""
import numpy as np
from numpy.linalg import norm
rad = np.pi/180

test = 0


def rotate2(vec, axis, theta):
    """
    Rotate vec around axis by theta radians.

    Parameters
    ----------
    vec : 1D or 2D np.ndarray (last axis length = 3).
    axis : 1D np.ndarray (last axis length = 3).
    theta : scalar or array-like.

    Returns
    -------
    rotated vec, shape = len(theta) * vec.shape
    """
    vec, axis = np.squeeze(vec), np.squeeze(axis)
    if (vec.shape[-1] != 3) or (axis.shape[-1] != 3):
        raise Exception("vectors' last axis length must = 3")

    # Determine if multiple vectors, multiple angles or both are input: act accordingly
    if (not len(vec.shape)>1) and (not hasattr(theta, '__iter__')):
        a, w = np.cos(-theta/2), -axis*np.sin(-theta/2)
        return vec + 2*a*np.cross(w, vec) + 2*np.cross(w, np.cross(w, vec))

    if len(vec.shape)>1 and hasattr(theta, '__iter__'):
        theta = -np.squeeze(theta)
        a, w = np.cos(theta/2)[:,None], -axis*np.sin(theta/2)[:,None]
        return np.squeeze([v + 2*a*np.cross(w, v) + 2*np.cross(w, np.cross(w, v)) for v in vec])

    if (not len(vec.shape)>1) and hasattr(theta, '__iter__'):
        theta = -np.squeeze(theta)
        a, w = np.cos(theta/2)[:,None], -axis*np.sin(theta/2)[:,None]
        return vec + 2*a*np.cross(w, vec) + 2*np.cross(w, np.cross(w, vec))

    if (len(vec.shape)==1) and (not hasattr(theta, '__iter__')):
        a, w = np.cos(-theta/2), -axis*np.sin(-theta/2)
        return vec + 2*a*np.cross(w, vec) + 2*np.cross(w, np.cross(w, vec))

    print("Input dimensions invalid")


if (__name__ == '__main__') and test:
    print(rotate2([1,0,0], [0,0,1], 45*rad))
    print(rotate2([1,0,0], [0,0,1], np.linspace(0,180,10)*rad))
    print(rotate2([[1,0,0],[0.7071,0.7071,0]], [0,0,1], 45*rad))
    print(rotate2([[1,0,0],[0.7071,0.7071,0]], [0,0,1], [20*rad,40*rad]))
    print(rotate2([1,1,0], [0,1,0], 45*rad))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# Get angle between vectors #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_ang(v1, v2):
    """
    Get the angle between two vectors.

    Parameters
    ----------
    v1 : array-like (last axis length = 3).
    v2 : array-like (last axis length = 3).

    Returns
    -------
    array-like : angles between vectors in v1 and v2. Dims = (len(v1), len(v2)).
    """
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    if (not len(v1.shape)) or (not len(v2.shape)):
        raise Exception(f"v1 and v2 must be vectors (or a list of vectors) with length 3\nv1 = {v1}\tv2 = {v2}")
    if (not v1.shape[-1]) or (not v2.shape[-1]):
        raise Exception(f"v1 and v2 must be vectors (or a list of vectors) with length 3\nv1 = {v1}\tv2 = {v2}")

    if (len(v2.shape) > 1):
        return np.arccos( np.dot(v1, v2.T) / ( norm(v1, axis=-1) * norm(v2, axis=-1)[:,None] + 1e-20 ).T )

    return np.arccos( np.dot(v1, v2.T) / ( norm(v1, axis=-1) * norm(v2, axis=-1) + 1e-20 ).T )

    return np.arccos( np.dot(v1, v2.T) / ( norm(v1) * norm(v2) + 1e-20 ) )


if (__name__ == '__main__') and test:
    vecs = [ [1,1,0], [-1,1,0], [-1,-1,0], [1,-1,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0] ]
    for vec in vecs:
        ang = get_ang([1,0,0], vec)
        print(f'{vec}:\t{ang/rad:.0f}')

    # Test dimensions
    # v1_ndim = 1, v2_ndim = 1
    v1, v2 = [1,1,1], [0,0,1]
    print(get_ang(v1, v2)/rad)
    # v1_ndim = 2, v2_ndim = 1
    v1, v2 = [[1,0,0],[0,1,0],[0,0,1], [1,1,1]], [0,0,1]
    print(get_ang(v1, v2)/rad)
    # v1_ndim = 2, v2_ndim = 2
    v1, v2 = [[1,0,0],[0,1,0],[0,0,1], [1,1,1]], [[0,0,1], [0,0,2]]
    print(get_ang(v1, v2)/rad)
    # v1_ndim = 2, v2_ndim = 1
    v1, v2 = np.array([[1,0,0],[0,1,0],[0,0,1], [1,1,1]]), 0  # Raise exception
    print(get_ang(v1, v2)/rad)


