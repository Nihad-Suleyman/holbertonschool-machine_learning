#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx (int): Number of input features.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the neuron's weights."""
        return self.__W

    @property
    def b(self):
        """Return the neuron's bias."""
        return self.__b

    @property
    def A(self):
        """Return the neuron's activated output."""
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

    def cost(self, Y, A):
        """Calculate the logistic regression cost.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Predicted probabilities with shape (1, m).

        Returns:
            float: Logistic regression cost.
        """
        m = Y.shape[1]

        loss = (
            Y * np.log(A)
            + (1 - Y) * np.log(1.0000001 - A)
        )

        return -np.sum(loss) / m
