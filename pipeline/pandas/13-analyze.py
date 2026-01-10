#!/usr/bin/env python3
"""we will exclude the timestamp"""


def analyze(df):
    """just one line of code which is describing"""
    return df.drop(columns=['Timestamp']).describe()
