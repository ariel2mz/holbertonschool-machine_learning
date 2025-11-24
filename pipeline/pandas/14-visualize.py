#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")
df["Close"] = df["Close"].fillna(method="ffill")

for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])

df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

transformed_df = df.copy()

daily = df["2017":].resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

plt.figure(figsize=(12, 6))
plt.plot(daily.index, daily["Close"])
plt.title("Daily Close Price (2017+)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.tight_layout()
plt.show()

print(daily)
