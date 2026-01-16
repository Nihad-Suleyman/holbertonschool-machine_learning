#!/usr/bin/env python3
"""now we will use calculus knowledge in python"""


def summation_i_squared(n):
    """we will sum the square of numbers"""
    if !isnum(n):
        return None
    cnt = 0
    for i in range(n):
        cnt += i**2
    return cnt
