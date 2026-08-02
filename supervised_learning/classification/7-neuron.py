#!/usr/bin/env python3
"""Defines a single neuron for binary classification."""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
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
        """Calculate the neuron's forward propagation."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """Calculate the logistic regression cost."""
        m = Y.shape[1]

        loss = (
            Y * np.log(A)
            + (1 - Y) * np.log(1.0000001 - A)
        )

        return -np.sum(loss) / m

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]

        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron using gradient descent."""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")

            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        iteration_values = []
        cost_values = []

        for iteration in range(iterations + 1):
            A = self.forward_prop(X)

            should_report = (
                (verbose or graph)
                and (
                    iteration % step == 0
                    or iteration == iterations
                )
            )

            if should_report:
                current_cost = self.cost(Y, A)

                if verbose:
                    print(
                        "Cost after {} iterations: {}".format(
                            iteration,
                            current_cost
                        )
                    )

                if graph:
                    iteration_values.append(iteration)
                    cost_values.append(current_cost)

            if iteration < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(iteration_values, cost_values, "b")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
