#!/usr/bin/env python3
"""Now we will look at definiteness of matrices."""
import numpy as np


def definiteness(matrix):
    """This funnction is for checking definiteness"""
    if isinstance(matrix, np.ndarray) is False:
        raise TypeError('matrix must be a numpy.ndarray')
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return None
    for i in range(len(matrix)):
        if isinstance(matrix[i], np.ndarray) is False:
            raise TypeError('matrix must be a numpy.ndarray')
        if len(matrix) != len(matrix[i]):
            return None
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != matrix[j][i]:
                    return None
        arr = np.array(matrix)
        eigvals = np.linalg.eigvals(arr)
        if np.all(eigvals > 0):
            return 'Positive definite'
        elif np.all(eigvals >= 0):
            return 'Positive semi-definite'
        elif np.all(eigvals < 0):
            return 'Negative definite'
        elif np.all(eigvals <= 0):
            return 'Negative semi-definite'
        else:
            return 'Indefinite'
