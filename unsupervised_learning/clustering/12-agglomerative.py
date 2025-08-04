#!/usr/bin/env python3
"""sadasdasdasdas"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    sadasdasdasdas
    """
    Z = sch.linkage(X, method='ward')
    clss = sch.fcluster(Z, t=dist, criterion='distance')

    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(Z, color_threshold=dist, above_threshold_color='gray')
    plt.title('Dendrogram with Ward Linkage')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()

    return clss
