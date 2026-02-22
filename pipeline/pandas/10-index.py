#!/usr/bin/env python3
"""now we will try to mkae column an index"""


def index(df):
    """We will take transpose and then set index"""
    df['Timestamp'] = df['Timestamp'].T
    return df.set_index('Timestamp')
