#!/usr/bin/env python3
"""We will look at binomial distribution"""


class Binomial:
    """binomial class"""
    def __init__(self, data=None, n=1, p=0.5):
        """We will initialize the necessary terms"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(i for i in data) / len(data)
            p = mean / len(data)
            n = mean / p
        else:
            if n < 0:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
        self.n = round(n)
        self.p = p