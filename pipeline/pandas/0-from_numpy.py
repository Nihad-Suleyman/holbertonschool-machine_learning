#!/usr/bin/env python3

"""let's start with creating function"""
def from_numpy(array):
    import pandas as pd
    df = pd.DataFrame(array)
    df.columns = [chr(ord('A') + i) for i in range(array.shape[1])]
    return df
