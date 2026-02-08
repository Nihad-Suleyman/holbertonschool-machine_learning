#!/usr/bin/env python3
"""We will look at normal distribution"""


class Normal:
    """Normal distribution"""
    pi = 3.141592654
    e = 2.7182818285
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
        
        return (1 / ((2 * Normal.pi) ** 0.5 * self.stddev)) *\
            Normal.e ** (-1/2 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """Cumulative distribution function for a Normal distribution"""
        val = (x - self.mean) / (self.stddev * (2 ** (1 / 2)))
        erf1 = (2 / Normal.pi ** (1 / 2))
        erf2 = (val - (val ** 3) / 3 + (val ** 5) / 10 - (val ** 7) / 42
                + (val ** 9) / 216)
        cdf = (1 / 2) * (1 + erf1 * erf2)
        return cdf
