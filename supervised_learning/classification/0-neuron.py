#!/usr/bin/env python3
import numpy as np
"""
Decision Tree implementation module.

This module contains the implementation of a basic Decision
for supervised learning. It includes the definition of intern
and the overall tree structure. The tree supports
and count the number of nodes or leaf nodes.

Classes:
    Node: Represents an internal node in the decision tree.
    Leaf: Represents a leaf node (a terminal node with a value).
    Decision_Tree: The main decision tree class that manages tree structure.

Methods provided:
    - max_depth_below: Recursively determines the maximum depth.
    - count_nodes_below: Recursively counts nodes or leaves.
"""


class Neuron:
    """
    Represents an internal node in the decision tree.

    Attributes:
        feature (int): The feature index used for splitting.
        threshold (float): The threshold value for splitting.
        left_child (Node): The left child node.
        right_child (Node): The right child node.
        is_leaf (bool): True if node is a leaf.
        is_root (bool): True if node is the root.
        sub_population (array): The data subset at this node.
        depth (int): The depth of the node in the tree.
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
