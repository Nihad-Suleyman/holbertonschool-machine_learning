#!/usr/bin/env python3
"""sorting by value"""


def high(df):
    """we will sort in descending order by High"""
    df = df.sort_values(ascending=False, by='High')
    return df
