#!/usr/bin/env python3
"""
Functions to save and load Keras model weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a Keras model.

    Args:
        network: The Keras model whose weights should be saved.
        filename: Path of the file to save the weights to.
        save_format: Format to save the weights in ('keras' or 'h5').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads weights into a Keras model.

    Args:
        network: The Keras model to load weights into.
        filename: Path of the file containing the weights.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
