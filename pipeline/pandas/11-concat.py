#!/usr/bin/env python3
"""
asgsagsagsaa
"""


def concat(df1, df2):
    """
    klgsalfkaslf
    """

    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    df2f = df2[df2.index <= 1417411920]

    result = __import__('pandas').concat(
        [df2f, df1],
        keys=['bitstamp', 'coinbase']
    )

    return result
