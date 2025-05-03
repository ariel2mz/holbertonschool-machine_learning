#!/usr/bin/env python3
"""
sadsadsadsa dsadsa
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: The Keras model to train.
        data: numpy.ndarray of shape (m, nx) containing the input data.
        labels: one-hot numpy.ndarray of shape (m, classes) containing the labels.
        batch_size: Size of each mini-batch for gradient descent.
        epochs: Number of passes through the data.
        verbose: Boolean to control the verbosity of training output.
        shuffle: Boolean that determines whether to shuffle the data every epoch.

    Returns:
        History object generated after training the model.
    """
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)