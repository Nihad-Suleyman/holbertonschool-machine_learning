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
            var = sum((x - mean) ** 2 / len(data) for x in data)
            p = 1 - var / mean
            n = round(mean / p)
            p = mean / n
        else:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
        self.n = round(n)
        self.p = float(p)

    def pmf(self, k):
        """probability mass function"""
        factn, factk, factnk = 1, 1, 1
        for i in range(1, self.n + 1):
            factn *= i
        for i in range(1, k + 1):
            factk *= k
        for i in range(1, self.n + 1 - k):
            factnk *= i
        return factn / (factk * factnk) * self.p ** k * (1 - self.p) ** (self.n - k)