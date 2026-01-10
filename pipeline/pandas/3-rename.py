#!/usr/bin/env python3
"""now let's do with using filename"""


import pandas as pd


def from_file(filename, delimiter):
    """create a DataFrame from a file"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
