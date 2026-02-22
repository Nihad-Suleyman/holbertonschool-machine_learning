#!/usr/bin/env python3
"""Calculates specificity for each class"""


import numpy as np


def specificity(confusion):
    """
    Calculates specificity for each class
    """
    total = np.sum(confusion)
    true_positives = np.diag(confusion)
    row_sum = np.sum(confusion, axis=1)
    col_sum = np.sum(confusion, axis=0)
    false_positives = col_sum - true_positives
    false_negatives = row_sum - true_positives
    true_negatives = total - (true_positives + false_positives + false_negatives)

    return true_negatives / (true_negatives + false_positives)