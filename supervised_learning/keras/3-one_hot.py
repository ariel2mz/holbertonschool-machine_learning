#!/usr/bin/env python3
import numpy as np

def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    
    Parameters:
    - labels: np.ndarray of shape (m,) containing the class labels
    - classes: total number of classes. If None, inferred from labels
    
    Returns:
    - A one-hot encoded matrix of shape (m, classes)
    """
    if classes is None:
        classes = np.max(labels) + 1
    return np.eye(classes)[labels]