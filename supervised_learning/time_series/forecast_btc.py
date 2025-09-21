#!/usr/bin/env python3
"""
asdsadsas
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os


def build_model(seq_len, n_features):
    """sadasdsadassa"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main(args):
    """
    sadsadsa
    """
    data = np.load(args.data)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_model(seq_len, n_features)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(args.model_out, save_best_only=True)
    ]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1000).batch(args.batch).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(args.batch)

    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(f"Test MSE: {test_loss:.6f}, Test MAE: {test_mae:.6f}")

    base = os.path.splitext(args.data)[0]
    target_scaler = joblib.load(base + "_target_scaler.pkl")

    y_pred = model.predict(test_ds)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test)

    np.savez_compressed("predictions.npz",
                        y_true=y_test_inv,
                        y_pred=y_pred_inv)
