#!/usr/bin/env python3
"""it is time to look at swtiching to numpy"""


def array(df):
    """convert DataFrame to array"""
    df = pd.DataFrame(df)
    df.columns = ['High', 'Close']
    DataFrame = df.tail(10)
    DataFrame = DataFrame[["High", "Close"]]
    ndarray = DataFrame.to_numpy()
    return ndarray
