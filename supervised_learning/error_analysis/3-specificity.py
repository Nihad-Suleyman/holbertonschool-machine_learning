#!/usr/bin/env python3
"""Calculates specificity for each class"""


import numpy as np


def specificity(confusion):
    """Calculates specificity for each class"""
    total = np.sum(confusion)
    tp = np.diag(confusion)
    row_sum = np.sum(confusion, axis=1)
    col_sum = np.sum(confusion, axis=0)

    fp = col_sum - tp
    fn = row_sum - tp
    tn = total - (tp + fp + fn)

    return tn / (tn + fp)