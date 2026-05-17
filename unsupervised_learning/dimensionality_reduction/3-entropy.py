#!/usr/bin/env python3
"""Entropy module."""
import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities relative to a data point.
    """
    beta = float(beta)

    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)

    Hi = np.log2(sum_Pi) + beta * np.sum(Di * Pi) / (sum_Pi * np.log(2))
    Pi = Pi / sum_Pi

    return Hi, Pi
