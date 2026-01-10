#!/usr/bin/env python3
"""general change using iloc"""


def slice(df):
    """we will now look at one type of slicing in pandas"""
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
