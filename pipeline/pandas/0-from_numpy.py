#!/usr/bin/env python3
"""let's create dataframe using the n dimensional array"""


import pandas as pd

def from_numpy(array):
    """let's assign our df"""
    df = pd.DataFrame(array)
    df.columns = [chr(ord('A') + i) for i in range(array.shape[1])]
    return df
