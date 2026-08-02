#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def one_hot_decode(one_hot):
    """One-hot decode"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    if one_hot.size == 0:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    if not np.all(np.sum(one_hot, axis=0) == 1):
        return None
    return np.argmax(one_hot, axis=0)
