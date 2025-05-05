#!/usr/bin/env python3
"""
Function to make predictions with a Keras model.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a Keras neural network.

    Args:
        network: The Keras model to make the prediction with.
        data: The input data to predict.
        verbose: Boolean to control output verbosity during prediction.

    Returns:
        The prediction for the input data (as a NumPy array).
    """
    return network.predict(data, verbose=verbose)
