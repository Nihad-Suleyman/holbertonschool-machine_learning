#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
import matplotlib.pyplot as plt


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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.__L):
            if type(layers[lay]) is not int or layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")

            if lay == 0:
                prev = nx
            else:
                prev = layers[lay - 1]
            self.__weights['W{}'.format(lay + 1)] = (
                np.random.randn(layers[lay], prev) * np.sqrt(2 / prev)
            )

            self.__weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))

    @property
    def weights(self):
        return self.__weights

    @property
    def cache(self):
        return self.__cache

    @property
    def L(self):
        return self.__L

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for lay in range(self.__L):
            W = self.__weights['W{}'.format(lay + 1)]
            b = self.__weights['b{}'.format(lay + 1)]
            A_prev = self.__cache['A{}'.format(lay)]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A{}'.format(lay + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        m_loss = np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        cost = (1 / m) * (-m_loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the cost function"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            a = 'A' + str(i + 1)
            a_0 = 'A' + str(i)
            A = cache[a]
            A_0 = cache[a_0]
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[w]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[w]
            dw = np.matmul(A_0, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[w] = self.__weights[w] - alpha * dw.T
            self.__weights[b] = self.__weights[b] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the model"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            should_report = (
                (verbose or graph)
                and (i % step == 0 or i == iterations)
            )
            if should_report:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    temp_cost.append(cost)
                    temp_iterations.append(i)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(temp_iterations, temp_cost, "b")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()

        return self.evaluate(X, Y)
