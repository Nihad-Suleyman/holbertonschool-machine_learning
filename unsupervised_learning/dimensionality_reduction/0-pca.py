#!/usr/bin/env python3
"""PCA module."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vt = np.linalg.svd(X)

    cumulative = np.cumsum(S) / np.sum(S)

    nd = np.argmax(cumulative >= var) + 1

    W = Vt[:nd].T

    return W
