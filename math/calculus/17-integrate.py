#!/usr/bin/env python3
"""Polynomial integration."""


def poly_integral(poly, C=0):
    """Return the integral of a polynomial represented by a list of coefficients."""
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None
    integ = [C]
    for i, coeff in enumerate(poly):
        val = coeff / (i + 1)
        if val.is_integer():
            val = int(val)
        integ.append(val)
    while len(integ) > 1 and integ[-1] == 0:
        integ.pop()
    return integ
