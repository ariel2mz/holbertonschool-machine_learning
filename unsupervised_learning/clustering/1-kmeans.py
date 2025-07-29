#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    return centroids


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering."""
    # Input validation
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    # Initialize centroids (first use of np.random.uniform)
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for _ in range(iterations):
        # Assign clusters (no loops)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids (one loop)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[clss == j]
            if cluster_points.size > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize centroid (second use of np.random.uniform)
                new_centroids[j] = np.random.uniform(low=min_vals, high=max_vals)

        # Check for convergence
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clss