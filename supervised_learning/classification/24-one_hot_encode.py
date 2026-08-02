#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def one_hot_encode(Y, classes):
    """One-hot encode Y"""
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if type(classes) is not int or classes < 1 or Y.size == 0:
        return None
    if np.any(Y < 0) or classes <= np.amax(Y):
        return None

    ohe = np.zeros((classes, len(Y)))
    ohe[Y, np.arange(len(Y))] = 1
    return ohe
