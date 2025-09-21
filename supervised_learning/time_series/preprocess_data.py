#!/usr/bin/env python3
"""
sdasdsa
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_and_merge(files):
    """cargar las cosa"""
    dfs = []
    for f in files:
        df = pd.read_csv(f)

        df = df[['Timestamp', 'Open', 'High', 'Low', 'Close',
                 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.set_index('Timestamp')
        dfs.append(df)

    # merge y fill out 
    full = pd.concat(dfs).groupby('Timestamp').mean().sort_index()
    full = full.resample('1T').mean()
    full = full.ffill().bfill()
    return full


def create_sequences(data, target, seq_len, horizon):
    """sadasdas"""
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len+horizon-1])
    return np.array(X), np.array(y).reshape(-1, 1)


def main(args):
    """
    asdasdsa
    """
    df = load_and_merge(args.inputs)
    features = df[['Open', 'High', 'Low', 'Close',
                   'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]
    target = df[['Close']]

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    feat_scaler = StandardScaler()
    targ_scaler = StandardScaler()

    features_train = features.iloc[:train_end]
    target_train = target.iloc[:train_end]

    feat_scaler.fit(features_train)
    targ_scaler.fit(target_train)

    features_scaled = feat_scaler.transform(features)
    target_scaled = targ_scaler.transform(target)

    X, y = create_sequences(features_scaled, target_scaled,
                            args.seq_len, args.horizon)

    n_seq = len(X)
    train_end = int(n_seq * 0.8)
    val_end = int(n_seq * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    np.savez_compressed(args.out,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)

    joblib.dump(feat_scaler, os.path.splitext(args.out)[0] + "_feature_scaler.pkl")
    joblib.dump(targ_scaler, os.path.splitext(args.out)[0] + "_target_scaler.pkl")

