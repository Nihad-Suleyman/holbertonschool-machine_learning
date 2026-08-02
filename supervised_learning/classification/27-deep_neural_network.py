#!/usr/bin/env python3
"""Script to train model and plotting
    DNN
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Class Deep Neural Network
    """

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
        for lay in range(self.L):
            if layers[lay] < 1 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        """Getter attr"""
        return self.__L

    @property
    def cache(self):
        """Getter attr"""
        return self.__cache

    @property
    def weights(self):
        """Getter attr"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation function"""
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            if lay == self.__L - 1:
                exponentials = np.exp(
                    Z - np.max(Z, axis=0, keepdims=True)
                )
                self.__cache["A" + str(lay + 1)] = (
                    exponentials
                    / np.sum(exponentials, axis=0, keepdims=True)
                )
            else:
                cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """Cost function"""

        m = Y.shape[1]
        C = (-1 / m) * np.sum(Y * np.log(A))
        return C

    def evaluate(self, X, Y):
        """Evaluation function"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient"""

        m = Y.shape[1]
        weights = self.__weights.copy()
        A2 = self.__cache["A" + str(self.__L - 1)]
        A3 = self.__cache["A" + str(self.__L)]
        W3 = weights["W" + str(self.__L)]
        b3 = weights["b" + str(self.__L)]
        dZ_input = {}
        dZ3 = A3 - Y
        dZ_input["dz" + str(self.__L)] = dZ3
        dW3 = (1 / m) * np.matmul(A2, dZ3.T)
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        self.__weights["W" + str(self.__L)] = W3 - (alpha * dW3).T
        self.__weights["b" + str(self.__L)] = b3 - (alpha * db3)

        for lay in range(self.__L - 1, 0, -1):
            cache = self.__cache
            Aa = cache["A" + str(lay)]
            Ap = cache["A" + str(lay - 1)]
            Wa = weights["W" + str(lay)]
            Wn = weights["W" + str(lay + 1)]
            ba = weights["b" + str(lay)]
            dZ1 = np.matmul(Wn.T, dZ_input["dz" + str(lay + 1)])
            dZ2 = Aa * (1 - Aa)
            dZ = dZ1 * dZ2
            dW = (1 / m) * np.matmul(Ap, dZ.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ_input["dz" + str(lay)] = dZ
            self.__weights["W" + str(lay)] = Wa - (alpha * dW).T
            self.__weights["b" + str(lay)] = ba - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Training"""
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
                raise ValueError("step must be positive and <= iterations")

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            should_report = (
                (verbose or graph)
                and (i % step == 0 or i == iterations)
            )
            if should_report:
                cost = self.cost(Y, self.__cache["A" + str(self.__L)])
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    temp_cost.append(cost)
                    temp_iterations.append(i)
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saving"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as model_file:
            pickle.dump(self, model_file)

    @staticmethod
    def load(filename):
        """Loading"""
        try:
            with open(filename, 'rb') as model_file:
                return pickle.load(model_file)
        except FileNotFoundError:
            return None
