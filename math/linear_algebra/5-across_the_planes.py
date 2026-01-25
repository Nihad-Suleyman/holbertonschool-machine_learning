#!/usr/bin/env python3
"""Now we will find the sum of 2d Arrays."""


def add_matrices2D(mat1, mat2):
    """Similar to the previous tasks."""
    if len(mat1) != len(mat2):
        return None
    new = []
    for j in range(len(mat1)):
        rows = []
        for i in range(len(mat1[0])):
            rows.append(mat1[j][i] + mat2[j][i])
        new.append(rows)
    return new
