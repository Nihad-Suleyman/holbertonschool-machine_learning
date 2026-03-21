#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization
    """
    m = Y.shape[1]
    dz = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        dW = (np.matmul(dz, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m
        if i > 1:
            A_prev_act = cache["A{}".format(i - 1)]
            dz = np.matmul(W.T, dz) * (1 - A_prev_act ** 2)
        weights["W{}".format(i)] = W - alpha * dW
        weights["b{}".format(i)] = b - alpha * db
