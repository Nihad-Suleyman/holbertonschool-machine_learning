#!/usr/bin/env python3
"""Gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent
    """
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if i > 1:
            dA = np.matmul(W.T, dZ)
            dA = dA * cache["D{}".format(i - 1)]
            dA = dA / keep_prob
            A = cache["A{}".format(i - 1)]
            dZ_next = dA * (1 - A ** 2)
        weights["W{}".format(i)] = W - alpha * dW
        weights["b{}".format(i)] = b - alpha * db
        if i > 1:
            dZ = dZ_next
