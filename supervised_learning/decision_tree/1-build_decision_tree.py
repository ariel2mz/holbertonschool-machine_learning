#!/usr/bin/env python3
"""
Decision Tree implementation module.

This module contains the implementation of a basic Decision Tree data structure
for supervised learning. It includes the definition of internal nodes, leaf nodes,
and the overall tree structure. The tree supports functionality to compute its depth
and count the number of nodes or leaf nodes.

Classes:
    Node: Represents an internal node in the decision tree.
    Leaf: Represents a leaf node (a terminal node with a value).
    Decision_Tree: The main decision tree class that manages tree structure.

Methods provided:
    - max_depth_below: Recursively determines the maximum depth.
    - count_nodes_below: Recursively counts nodes or leaves.
"""

import numpy as np


class Node:
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
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node object."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively returns the maximum depth below the current node.

        Returns:
            int: Maximum depth of the subtree rooted at this node.
        """
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = 0
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = 0
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Recursively counts nodes in the subtree.

        Args:
            only_leaves (bool): If True, only leaf nodes are counted.

        Returns:
            int: Number of nodes or leaf nodes below this node.
        """
        count = 0
        if not only_leaves:
            count += 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count


class Leaf(Node):
    """
    Represents a leaf node in the decision tree.

    Attributes:
        value (any): The output value stored at the leaf.
        depth (int): The depth of the leaf node.
    """
    def __init__(self, value, depth=None):
        """Initialize a Leaf node."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since this is a leaf node.

        Args:
            only_leaves (bool): Unused here.

        Returns:
            int: 1
        """
        return 1


class Decision_Tree:
    """
    Main class for building and managing a Decision Tree.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum samples required to split a node.
        seed (int): Random seed for reproducibility.
        split_criterion (str): Criterion for splitting (e.g., "random").
        root (Node): Root node of the tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the Decision Tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Get the maximum depth of the tree.

        Returns:
            int: Depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Total number of (leaf) nodes.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
