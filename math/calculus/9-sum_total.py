#!/usr/bin/env python3
"""Compute sum_{i=1}^n i^2 without loops."""

def summation_i_squared(n):
    """Return sum of squares from 1 to n, or None if n is invalid."""
    if not isinstance(n, int) or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
