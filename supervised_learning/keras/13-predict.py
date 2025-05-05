#!/usr/bin/env python3
"""
Function to make predictions with a Keras model.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    predice.

    Args:
        network: modelo.
        data: la data mano vpi.
        verbose: boleano d iporimir!.

    Returns:
        array d data.
    """
    return network.predict(data, verbose=verbose)
