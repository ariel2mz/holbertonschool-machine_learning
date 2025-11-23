#!/usr/bin/env python3
"""
asgsagsagsaa
"""


def fill(df):
    """
    asfasfsa
    """
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
