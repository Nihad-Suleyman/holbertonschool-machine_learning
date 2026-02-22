#!/usr/bin/env python3
"""we will now use concatenate"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """we will use as previous"""
    return np.concatenate((mat1, mat2), axis=axis)
