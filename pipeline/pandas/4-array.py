#!/usr/bin/env python3
"""it is time to look at swtiching to numpy"""


def array(df):
    """convert DataFrame to array"""
    return df[["High", "Close"]].tail(10).to_numpy()
