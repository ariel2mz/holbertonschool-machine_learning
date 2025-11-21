#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import pandas as pd


def rename(df):
    """
    sfsafsafsa
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime", "Close"]]

    return df
