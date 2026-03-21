#!/usr/bin/env python3
"""Forward propagation with dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        A_prev = cache["A{}".format(i - 1)]
        Z = np.matmul(W, A_prev) + b
        if i == L:
            Z_shift = Z - np.max(Z, axis=0, keepdims=True)
            exp_Z = np.exp(Z_shift)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache["A{}".format(i)] = A
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob
            cache["D{}".format(i)] = D
            cache["A{}".format(i)] = A

    return cache