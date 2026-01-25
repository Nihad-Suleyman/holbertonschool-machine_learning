#!/usr/bin/env python3
"""Now we will add up 2 arrays elementwise"""


def add_arrays(arr1, arr2):
    """This is our specific funtion, we will just add."""
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
