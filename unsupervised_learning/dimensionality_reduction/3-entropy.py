#!/usr/bin/env python3
"""Entropy module."""
import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities relative to a data point.
    """
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    Pi = P / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
