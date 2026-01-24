#!/usr/bin/env python3
"""We will look at how to print the shapes of matrix"""


def matrix_shape(matrix):
    """Okay, so we will look at how to go deep down of matrix size"""
    if not isinstance(matrix, list):
        return []
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
