#!/usr/bin/env python3
"""
asgsagsagsaa
"""


def flip_switch(df):
    """
    kasjfjasjkfjkasfjk
    """
    if "Timestamp" in df.columns:
        dfs = df.sort_values(by="Timestamp", ascending=False)
    else:
        dfs = df.sort_index(ascending=False)
    dft = dfs.transpose()
    return dft
