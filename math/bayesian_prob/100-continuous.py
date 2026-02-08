#!/usr/bin/env python3
"""1 advanced task"""
from scipy.special import betainc


def posterior(x, n, p1, p2):
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float):
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float):
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    a = x + 1
    b = n - x + 1
    cdf2 = betainc(a, b, p2)
    cdf1 = betainc(a, b, p1)
    return cdf2 - cdf1
    