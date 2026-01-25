#!/usr/bin/env python3
"""We will now work at elementwise method"""


def np_elementwise(mat1, mat2):
    """like before, but this time add and others"""
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    return add, sub, mul, div
