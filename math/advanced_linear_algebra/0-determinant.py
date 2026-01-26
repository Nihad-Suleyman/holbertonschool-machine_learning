#!/usr/bin/env python3
"""Now we will calculate the MIGHTY determinant"""


def determinant(matrix):
    """We will find the determinant, taking into conseideration
    some exceptions"""
    if isinstance(matrix, list) == False:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if isinstance(matrix[i], list) == False:
            raise TypeError('matrix must be a list of lists')
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    if matrix == [[]]:
        return 1
    elif len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    elif len(matrix) == 2 and len(matrix[0]) == len(matrix[1]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    cnt = 0
    for j in range(len(matrix)):
        minor = []
        for row in matrix[1:]:
            minor.append(row[:j] + row[j+1:])
        cnt += matrix[0][j] * (-1) ** j * determinant(minor)
    return cnt