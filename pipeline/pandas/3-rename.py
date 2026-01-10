#!/usr/bin/env python3
"""now let's do with using filename"""


import pandas as pd


def rename(df):
    """create a DataFrame from a file"""
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]

    return df
