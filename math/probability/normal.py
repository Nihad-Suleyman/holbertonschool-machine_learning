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
        """Cumulative distribution function for a Normal distribution"""
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        e = 2.718281828459045
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        sign = 1
        if z < 0:
            sign = -1
        z = abs(z)
        t = 1 / (1 + p * z)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf = sign * (1 - poly * (e ** (-z * z)))
        return 0.5 * (1 + erf)
