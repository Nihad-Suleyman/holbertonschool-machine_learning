#!/usr/bin/env python3
"""now an advanced task"""


def np_slice(matrix, axes={}):
    """we are looking at slices"""
    slices = [slice(None)] * matrix.ndim
    for axis, values in axes.items():
        slices[axis] = slice(*values)
    return matrix[tuple(slices)]
