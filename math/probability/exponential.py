#!/usr/bin/env python3
"""We will look at the behaviour of exponential"""


class Exponential:
    """Exponential class"""
    def __init__(self, data=None, lambtha=1.):
        """We should initialize necessary terms"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            lambtha = 1 / (sum(i for i in data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        self.lambtha = float(lambtha)
