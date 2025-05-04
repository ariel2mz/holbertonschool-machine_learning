#!/usr/bin/env python3
"""
Functions to save and load a Keras model configuration in JSON format.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves the configuration of a Keras model in JSON format.

    Args:
        network: The Keras model whose configuration should be saved.
        filename: Path of the file to save the configuration to.

    Returns:
        None
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)
    return None


def load_config(filename):
    """
    Loads a Keras model from a JSON configuration file.

    Args:
        filename: Path of the JSON file with the model configuration.

    Returns:
        The Keras model (uncompiled and without weights).
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    model = K.models.model_from_json(json_config)
    return model
