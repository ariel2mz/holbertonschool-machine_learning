#!/usr/bin/env python3
"""
Function to test a Keras model.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    test.

    Args:
        network: el modelo
        data: data de entrenamiento
        labels: onehot encoded
        verbose: print si o no.

    Returns:
        perdida precision
    """
    return network.evaluate(data, labels, verbose=verbose)
