#!/usr/bin/env python3
"""Now we will look at how cofactors work"""


def determinant(matrix):
    """We will find the determinant, taking into conseideration
    some exceptions"""
    if isinstance(matrix, list) is False:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    for i in range(len(matrix)):
        if isinstance(matrix[i], list) is False:
            raise TypeError('matrix must be a list of lists')
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2 and len(matrix[0]) == len(matrix[1]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    cnt = 0
    for j in range(len(matrix)):
        minor = []
        for row in matrix[1:]:
            minor.append(row[:j] + row[j+1:])
        cnt += matrix[0][j] * (-1) ** j * determinant(minor)
    return cnt


def cofactor(matrix):
    """This time it is for cofactor"""
    if isinstance(matrix, list) is False:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    for i in range(len(matrix)):
        if isinstance(matrix[i], list) is False:
            raise TypeError('matrix must be a list of lists')
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]
    if len(matrix) == 2 and len(matrix[0]) == len(matrix[1]) == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]
    cnt = []
    n = len(matrix)
    for i in range(n):
        row = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r == i:
                    continue
                sub.append(matrix[r][:j] + matrix[r][j+1:])
            m = determinant(sub)
            row.append(((-1) ** (i + j)) * m)
        cnt.append(row)

    return cnt

