#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class NeuralNetwork:
    """A class that represents a neural network"""

    def __init__(self, nx, nodes):
        """Constructor for the neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get the weights of the first layer"""
        return self.__W1

    @property
    def W2(self):
        """Get the weights of the second layer"""
        return self.__W2

    @property
    def b1(self):
        """Get the biases of the first layer"""
        return self.__b1

    @property
    def b2(self):
        """Get the biases of the first layer"""
        return self.__b2

    @property
    def A2(self):
        """Get the activation function of the second layer"""
        return self.__A2

    @property
    def A1(self):
        """Get the activation function of the second layer"""
        return self.__A1

    def forward_prop(self, X):
        """Forward propagation"""
        z1 = self.__W1.dot(X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = self.__W2.dot(self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the network"""
        m = Y.shape[1]
        m_loss = np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        cost = (1 / m) * (-m_loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the cost of the network"""
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates the gradient descent"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
