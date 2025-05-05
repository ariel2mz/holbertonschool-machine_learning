#!/usr/bin/env python3
"""
Function to test a Keras model.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a Keras neural network.

    Args:
        network: The Keras model to test.
        data: Input data to test the model with.
        labels: One-hot encoded correct labels for the data.
        verbose: Boolean indicating if output should be printed during testing.

    Returns:
        A tuple (loss, accuracy) from evaluating the model.
    """
    return network.evaluate(data, labels, verbose=verbose)
