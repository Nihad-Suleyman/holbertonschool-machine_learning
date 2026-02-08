#!/usr/bin/env python3
"""We will look at normal distribution"""
import numpy as np

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
        
        # Using a precise constant for e
        e = 2.7182818285
        
        # Calculate z-score: (x - mean) / stddev
        z = (x - self.mean) / self.stddev
        
        # Save the sign of z for erf calculation
        sign = 1 if z >= 0 else -1
        abs_z = abs(z)
        
        # Abramowitz & Stegun approximation for erf(z)
        t = 1.0 / (1.0 + p * abs_z)
        poly = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5)
        
        # erf(z) = sign * (1 - poly * e^(-z^2))
        res_erf = sign * (1 - poly * (e ** -(abs_z**2)))
        
        # Normal CDF formula: 0.5 * (1 + erf(z / sqrt(2)))
        # NOTE: The standard Normal CDF relates to erf via z / sqrt(2)
        # However, the constants you chose are often used for a direct 
        # approximation of the CDF. Let's adjust to the standard erf route:
        
        val = (x - self.mean) / (self.stddev * (2**0.5))
        sign = 1 if val >= 0 else -1
        v = abs(val)
        t = 1.0 / (1.0 + p * v)
        poly = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5)
        erf_val = sign * (1 - poly * (e ** -(v**2)))
        
        return 0.5 * (1 + erf_val)
