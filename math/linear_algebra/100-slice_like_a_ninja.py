#!/usr/bin/env python3
"""now an advanced task"""
import numpy as np


def np_slice(matrix, axes={}):
    slices = [slice(None)] * matrix.ndim
    for axis, values in axes.items():
        slices[axis] = slice(*values)
    return matrix[tuple(slices)]
