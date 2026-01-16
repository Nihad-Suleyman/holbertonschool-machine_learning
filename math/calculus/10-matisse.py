#!/usr/bin/env python3
"""Computing the coefficents of a derrivative of polynomial."""

def poly_derivative(poly):
    """We will use list."""
    if not isinstance(poly, list) or len(poly) <= 0:
        return None
    new_poly = []
    for i in range(1, len(poly)):
        new_poly.append(i * poly[i])
    return new_poly

