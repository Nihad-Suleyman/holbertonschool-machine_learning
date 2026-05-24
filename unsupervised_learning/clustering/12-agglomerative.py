#!/usr/bin/env python3
"""Agglomerative clustering."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset."""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        dist,
        criterion='distance'
    )

    return clss
