#!/usr/bin/env python3
"""now we will flip the related data and sort it in ascendinng order"""


def flip_switch(df):
    """Docstring for flip_switch"""
    df = df.sort_index(ascending=False)
    df.columns, df.rows = df.rows, df.columns
    return df
