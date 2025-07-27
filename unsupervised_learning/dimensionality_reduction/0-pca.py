#!/usr/bin/env python3
"""sadsadsadasdsa"""
import numpy as np


def pca(X, var=0.95):
    """
    What is PCA?
    PCA stands for Principal Component Analysis.
    It's a way to reduce the number of features (dimensions) in your data
    while keeping as much important information as possible.

    Why reduce dimensions?
    Imagine you have data with many features (like height, weight, age, etc).
    Some of these features may be related or redundant.
    PCA helps you find a smaller set of features that still captures
    most of the patterns in your data.

    How does PCA work?
    Input: A dataset where each row is a data point.
    Centering: Subtract the mean from each feature (already done here).
    Covariance: Check how features change together.
    Eigenvectors: Find new directions (called components).
    Sort: Choose the directions that explain the most variance.
    Project: Keep only the directions that explain enough variance.
    """

    covm = np.cov(X, rowvar=False)
    eivals, eivecs = np.linalg.eigh(covm)

    sortedidx = np.argsort(eivals)[::-1]
    eivals = eivals[sortedidx]
    eivecs = eivecs[:, sortedidx]

    cuvar = np.cumsum(eivals) / np.sum(eivals)

    nd = np.searchsorted(cuvar, var) + 1
    W = eivecs[:, :nd]

    return W
