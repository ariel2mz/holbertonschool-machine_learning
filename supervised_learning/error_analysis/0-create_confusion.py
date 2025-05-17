#!/usr/bin/env python3
"""asdasdasdsadsa"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels (np.ndarray): Onehot encoded true labels
        logits (np.ndarray): Onehot encoded predicted labels

    Returns:
        np.ndarray: Confusion matrix of shape (classes, classes)
    """
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    conf = np.zeros((classes, classes), dtype=np.float64)

    for t, p in zip(true, pred):
        conf[t, p] += 1

    return conf
