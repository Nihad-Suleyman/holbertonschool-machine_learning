#!/usr/bin/env python3
"""Normalizes an unactivated output using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_batch = gamma * Z_norm + beta

    return Z_batch
