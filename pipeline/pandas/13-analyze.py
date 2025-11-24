#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import pandas as pd


def analyze(df):
    """
    kajfksafjlaksfjaslfk
    """

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    stats = df.describe()

    return stats
