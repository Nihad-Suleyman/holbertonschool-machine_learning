#!/usr/bin/env python3

import pandas as pd
"""let's start with creating function"""
def from_numpy(array):
    """let's assign our df"""
    df = pd.DataFrame(array)
    df.columns = [chr(ord('A') + i) for i in range(array.shape[1])]
    return df
