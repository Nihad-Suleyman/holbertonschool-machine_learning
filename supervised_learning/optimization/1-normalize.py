#!/usr/bin/env python3
"""Normalizes a matrix"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix
    """
    return (X - m) / s
