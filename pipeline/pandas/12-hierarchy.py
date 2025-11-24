#!/usr/bin/env python3
"""
hierarchy
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    asfgasfasfsafas
    """

    df1 = index(df1)
    df2 = index(df2)

    start = 1417411980
    end = 1417417980

    df1_slice = df1.loc[start:end]
    df2_slice = df2.loc[start:end]

    combined = pd.concat(
        [df2_slice, df1_slice],
        keys=["bitstamp", "coinbase"]
    )

    if combined.index.names != ["Timestamp", None]:
        combined = combined.swaplevel(0, 1)
    combined = combined.sort_index(level=0)

    return combined
