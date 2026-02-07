#!/usr/bin/env python3
"""We will look at poisson distribution"""


class Poisson:
    """poisson class"""
    def __init__(self, data=None, lambtha=1.):
        """We will initialize the necessary terms"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            lambtha = sum(i for i in data) / len(data)
        else:
            if lambtha <= 0:
                raise ValueError('lambda must be a positive value')
            
        self.lambtha = float(lambtha)
        # fact = 1
        # for i in range(1, len(data)+1):
        #     fact *= data
        # return math.exp(-lambtha) * lambtha ** len(data) / fact