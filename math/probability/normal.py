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
        e = 2.718281828459045
        p = 0.2316419
        z = (x - self.mean) / self.stddev
        abs_z = abs(z)
        t = 1 / (1 + p * abs_z)
        d = 0.3989422804014327 * (e ** (-abs_z * abs_z / 2))
        poly = (
            0.319381530 * t
            - 0.356563782 * t**2
            + 1.781477937 * t**3
            - 1.821255978 * t**4
            + 1.330274429 * t**5
        )
        cdf_pos = 1 - d * poly
        if z < 0:
            return 1 - cdf_pos
        return cdf_pos
