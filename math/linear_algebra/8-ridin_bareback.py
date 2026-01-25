#!/usr/bin/env python3
"""Now we will multiply 2 matrices."""


def mat_mul(mat1, mat2):
    """We should multiiply one matrix with transpose"""
    if len(mat1[0]) != len(mat2):
        return None
    mat2t = list(zip(*mat2))
    return [[sum(a * b for a, b in zip(rows, cols)) for cols in mat2t] for rows in mat1]
