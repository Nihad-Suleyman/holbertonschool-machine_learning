#!/usr/bin/env python3
"""Entropy module."""
import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities relative to a data point.
    """
    Pi = np.exp(-Di * beta)

    sum_Pi = np.sum(Pi)

    if sum_Pi == 0:
        Pi = np.zeros_like(Pi)
        Hi = 0
    else:
        Hi = np.log2(sum_Pi) + beta * np.sum(Di * Pi) / (sum_Pi * np.log(2))
        Pi = Pi / sum_Pi

    return Hi, Pi
