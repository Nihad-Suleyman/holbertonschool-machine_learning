#!/usr/bin/env python3
"""We will look at poisson distribution"""
import math

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
                raise ValueError('lambtha must be a positive value')
        self.lambtha = float(lambtha)
        
    def pmf(self, k):
        """We will calculate the PMF of poisson"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return math.exp(-self.lambtha) * self.lambtha ** k / fact
