#!/usr/bin/env python3
"""it is time to look at swtiching to numpy"""


def array(df):
    """convert DataFrame to array"""
    return df.tail(10)[["High", "CLose"]].to_numpy()
