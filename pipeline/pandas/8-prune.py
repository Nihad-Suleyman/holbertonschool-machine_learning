#!/usr/bin/env python3
"""now we will look at null values and damage them :) """


def prune(df):
    """we are looking for null values of close"""
    return df.dropna(subset=['Close'], axis=0)
