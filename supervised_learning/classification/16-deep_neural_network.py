#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class DeepNeuralNetwork:
    """A class that does the trick"""

    def __init__(self, nx, layers):
        """Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for lay in range(self.L):
            if type(layers[lay]) is not int or layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")

            if lay == 0:
                prev = nx
            else:
                prev = layers[lay - 1]
            self.weights['W{}'.format(lay + 1)] = (
                np.random.randn(layers[lay], prev) * np.sqrt(2 / prev)
            )

            self.weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))
