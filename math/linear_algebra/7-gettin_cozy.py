#!/usr/bin/env python3
"""Now we will concatenate 2D arrays."""


def cat_matrices2D(mat1, mat2, axis=0):
    """We will do as previous but this time with axis"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        new = []
        for i in range(min(len(mat1), len(mat2))):
            new += [mat1[i] + mat2[i]]
        return new
