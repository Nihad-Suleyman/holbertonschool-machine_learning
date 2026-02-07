#!/usr/bin/env python3
"""Now we will look at definiteness of matrices."""
import numpy as np


def definiteness(matrix):
    """This funnction is for checking definiteness"""
    if isinstance(matrix, list) is False:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    for i in range(len(matrix)):
        if isinstance(matrix[i], list) is False:
            raise TypeError('matrix must be a list of lists')
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a non-empty square matrix')
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != matrix[j][i]:
                    raise ValueError('matrix must be symmetric')
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