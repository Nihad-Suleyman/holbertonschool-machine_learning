#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

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
        """Return the neuron's weights vector."""
        return self.__W

    @property
    def b(self):
        """Return the neuron's bias."""
        return self.__b

    @property
    def A(self):
        """Return the neuron's activated output."""
        return self.__A
