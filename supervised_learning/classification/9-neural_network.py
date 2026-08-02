#!/usr/bin/env python3
"""Defines a neural network with one hidden layer."""

import numpy as np


class NeuralNetwork:
    """Neural network performing binary classification."""

    def __init__(self, nx, nodes):
        """Initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return the hidden layer's weights."""
        return self.__W1

    @property
    def b1(self):
        """Return the hidden layer's bias."""
        return self.__b1

    @property
    def A1(self):
        """Return the hidden layer's activated output."""
        return self.__A1

    @property
    def W2(self):
        """Return the output neuron's weights."""
        return self.__W2

    @property
    def b2(self):
        """Return the output neuron's bias."""
        return self.__b2

    @property
    def A2(self):
        """Return the output neuron's activated output."""
        return self.__A2
