#!/usr/bin/env python 3
"""now let's do with using filename"""


import pandas as pd


def from_file(filename, delimiter=','):
    DataFrame = pd.read_csv(filename, sep=delimiter)
    return DataFrame
