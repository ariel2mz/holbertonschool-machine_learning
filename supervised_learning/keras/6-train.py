#!/usr/bin/env python3
"""
sadsadsa sadsa
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient.

    Args:
        network: The Keras model to train.
        data: numpy.ndarray of shape (m, nx) with the input data.
        labels: one-hot numpy.ndarray of shape (m, classes).
        batch_size: Size of each mini-batch.
        epochs: Number of training epochs.
        validation_data: Tuple (val_data, val_labels) for validatio.
        early_stopping: Boolean indicating whether to apply stopping.
        patience: Number of epochs to wait after no improvement.
        verbose: Controls verbosity of training.
        shuffle: Whether to shuffle data each epoch.

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        es_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(es_callback)

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
