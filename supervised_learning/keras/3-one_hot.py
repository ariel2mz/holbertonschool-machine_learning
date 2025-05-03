#!/usr/bin/env python3
import tensorflow.keras as K

def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix using Keras backend.

    Args:
        labels: a numpy array or tensor of integer labels
        classes: total number of classes (optional)

    Returns:
        A one-hot encoded matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
