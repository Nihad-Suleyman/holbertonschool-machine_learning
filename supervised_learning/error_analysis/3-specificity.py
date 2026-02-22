#!/usr/bin/env python3
"""Calculates the F1 score for each class"""

import numpy as np


def f1_score(confusion):
    """Calculates the F1 score for each class"""
    tp = np.diag(confusion)
    precision = tp / np.sum(confusion, axis=0)
    recall = tp / np.sum(confusion, axis=1)

    return 2 * (precision * recall) / (precision + recall)
