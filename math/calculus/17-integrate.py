#!/usr/bin/env python3
"""Now we will look at an integral problem"""


def poly_integral(poly, C=0):
    """We will return the list of coefficients of an integral"""
    if (not isinstance(poly, int)) or (not isinstance(C, int)):
        return None
    new_poly = [C]
    for i in range(1, len(poly)):
        new_poly.append(poly[i - 1] / (i - 1))
    return new_poly
