#!/usr/bin/env python3
"""Calculates precision for each class"""


import numpy as np


def precision(confusion):
    """
    Calculates precision for each class
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
