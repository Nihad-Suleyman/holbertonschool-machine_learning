#!/usr/bin/env python3
"""Defines a neuron that performs binary classification."""

import numpy as np


class Neuron:
    """Represents a single neuron for binary classification."""

    def __init__(self, nx):
        """Initialize a neuron.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights vector."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            numpy.ndarray: Activated output with shape (1, m).
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
