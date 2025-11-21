#!/usr/bin/env python3
"""
asgsagsagsaa
"""


def slice(df):
    """
    asfsafsaf
    """
    sliced = df[["High", "Low", "Close", "Volume_(BTC)"]]
    sliced = sliced.iloc[::60]

    return sliced
