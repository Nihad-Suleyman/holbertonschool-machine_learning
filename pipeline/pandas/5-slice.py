#!/usr/bin/env python3
"""general change using iloc"""


def slice(df):
    """we will now look at one type of slicing in pandas"""
    df.columns = ['High', 'Low', 'Close', 'Volume_BTC']
    arr = df.iloc[::60]
    return arr
