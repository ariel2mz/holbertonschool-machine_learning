#!/usr/bin/env python3
"""sadasdas"""
import sklearn.cluster


def kmeans(X, k):
    """
    sadasdsadsa
    """
    model = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
