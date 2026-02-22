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

    def pdf(self, x):
        """calculating probability density funtion"""
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        """Now the cumulative part"""
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e ** (-self.lambtha * x)
