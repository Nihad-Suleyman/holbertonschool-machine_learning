#!/usr/bin/env python3
"""Calculates sensitivity for each class"""


import numpy as np


def sensitivity(confusion):
    """Calculates sensitivity for each class"""

    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
