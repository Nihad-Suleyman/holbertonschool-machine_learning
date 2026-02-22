#!/usr/bin/env python3
"""Now we will concatenate 2D arrays."""


def cat_matrices2D(mat1, mat2, axis=0):
    """We will do as previous but this time with axis"""
    if axis not in (0, 1):
        return None
    if mat1 == [] or mat2 == []:
        return None
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new = []
        for i in range(min(len(mat1), len(mat2))):
            new += [mat1[i] + mat2[i]]
        return new
