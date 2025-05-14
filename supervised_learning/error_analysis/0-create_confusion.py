#!/usr/bin/env python3
"""asdasdasdsadsa"""

import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels (np.ndarray): One-hot encoded true labels, shape (m, classes)
        logits (np.ndarray): One-hot encoded predicted labels, shape (m, classes)

    Returns:
        np.ndarray: Confusion matrix of shape (classes, classes)
    """
    # Convert one-hot to class indices
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    conf = np.zeros((classes, classes), dtype=np.int64)

    for t, p in zip(true, pred):
        conf[t, p] += 1

    return conf
