#!/usr/bin/env python3
"""t-SNE module."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.
    """
    X = pca(X, idims)

    P = P_affinities(X, perplexity=perplexity)
    P *= 4

    n = X.shape[0]

    Y = np.random.randn(n, ndims)

    dY = np.zeros((n, ndims))
    iY = np.zeros((n, ndims))

    for i in range(iterations):
        Y = Y - np.mean(Y, axis=0)

        dY, Q = grads(Y, P)

        if i < 20:
            a = 0.5
        else:
            a = 0.8

        iY = a * iY - lr * dY
        Y += iY

        if (i + 1) == 100:
            P /= 4

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

    return Y
