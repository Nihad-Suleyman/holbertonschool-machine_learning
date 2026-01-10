#!/usr/bin/env python3
"""We have plemty opf work to do here"""


def fill(df):
    """first we will change close, and then depending on that we will change the others"""
    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df.ffill(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
