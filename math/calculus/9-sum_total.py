#!/usr/bin/env python3
"""now we will use calculus knowledge in python"""


def summation_i_squared(n):
    """we will sum the square of numbers"""
    if isinstance(n, int) == False:
        return None
    if n == 1:
        return 1
    return n * n + summation_i_squared(n-1)
