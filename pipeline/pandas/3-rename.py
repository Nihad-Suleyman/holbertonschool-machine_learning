#!/usr/bin/env python3
"""now let's do with using filename"""


import pandas as pd


def rename(df):
    """create a DataFrame from a file"""
    df = df.rename(columns={'Timestamp': 'DateTime'})
    df['DateTime'] = pd.to_datetime(df['DateTime'], unit='s')
    df = df[['DateTime', 'Close']]

    return df
