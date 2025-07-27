#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def pca(X, var=0.95):
    """
    sadadsada
    """

    Lval, Sval, Rval = np.linalg.svd(
        X, full_matrices=False)

    cumvar = np.cumsum(Sval) / np.sum(Sval)
    comps = np.argmax(cumvar >= var) + 1

    return Rval[:comps].T
