#!/usr/bin/env python3
"""We will try to get transpose of a matrix"""


def matrix_transpose(matrix):
    """docs"""
    row = len(matrix)
    col = len(matrix[0])
    new = []
    for j in range(col):
        rowlist = []
        for i in range(row):
            rowlist.append(matrix[i][j])
        new.append(rowlist)
    return new
            