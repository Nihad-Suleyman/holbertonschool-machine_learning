#!/usr/bin/env python3
"""PCA module."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    U, S, Vt = np.linalg.svd(X)

    variances = S ** 2
    total_variance = np.sum(variances)
    cumulative_variance = np.cumsum(variances) / total_variance

    nd = np.argmax(cumulative_variance >= var) + 1

    W = Vt[:nd].T

    return W
