#!/usr/bin/env python3
"""We will look at normal distribution"""


class Normal:
    """Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """first initialization step"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(i for i in data) / len(data)
            stddev = (sum((i - mean) ** 2 / len(data) for i in data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        self.mean = mean
        self.stddev = stddev

    def z_score(self, x):
        """z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x value"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """probability density funtion of Normal"""
        pi = 3.141592654
        e = 2.7182818285
        return (1 / ((2 * pi) ** 0.5 * self.stddev)) *\
            e ** (-1/2 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """cumulative function of Normal"""
        e = 2.7182818285
        t = 1 / (1 + 0.2316419 * abs(x))
        d = 0.3989423 * e(-x * x / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
        return 1 - p
