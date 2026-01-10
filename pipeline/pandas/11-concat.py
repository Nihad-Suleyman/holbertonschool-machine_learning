#!/usr/bin/env python3
"""now we will use concatination"""


import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """we will set indexes of both dfs and then concatenate"""
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')
    df2 = df2.loc[:1417411920]
    df = pd.concat([df2, df1], keys=['bitsmap', 'coinbase'])
    return df
