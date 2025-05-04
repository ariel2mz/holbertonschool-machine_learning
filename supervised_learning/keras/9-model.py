#!/usr/bin/env python3
"""
Functions to save and load a Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network: The Keras model to save.
        filename: Path of the file to save the model to.

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename: Path of the file to load the model from.

    Returns:
        The loaded Keras model.
    """
    return K.models.load_model(filename)
