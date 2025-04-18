#!/usr/bin/env python3
"""
Decision Tree Module

This module provides a basic implementation of a decision tree structure
with internal decision nodes (`Node`) and terminal leaf nodes (`Leaf`). 
It includes methods to compute the depth of the tree, count the number
of nodes or leaf nodes, and represent the tree structure visually.

Classes:
    - Node
    - Leaf
    - Decision_Tree
"""

import numpy as np


class Node:
    """
    Represents an internal node in the decision tree.

    Attributes:
        feature (int): The index of the feature used for splitting.
        threshold (float): The threshold used for the split.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_leaf (bool): Whether this node is a leaf.
        is_root (bool): Whether this node is the root.
        sub_population (array): Optional population data at this node.
        depth (int): Depth of the node within the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initialize a Node object.

        Args:
            feature (int): Feature index to split on.
            threshold (float): Threshold value to split.
            left_child (Node): Left child node.
            right_child (Node): Right child node.
            is_root (bool): Whether the node is the root.
            depth (int): Depth of this node in the tree.
        """
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
        Compute the maximum depth below this node.

        Returns:
            int: The maximum depth in the subtree.
        """
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below() if self.left_child else 0
        right_depth = self.right_child.max_depth_below() if self.right_child else 0
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes in the subtree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.

        Returns:
            int: The number of nodes or leaves below this node.
        """
        if self.is_leaf:
            return 1
        if only_leaves:
            left_leaves = self.left_child.count_nodes_below(True) if self.left_child else 0
            right_leaves = self.right_child.count_nodes_below(True) if self.right_child else 0
            return left_leaves + right_leaves
        else:
            left_nodes = self.left_child.count_nodes_below() if self.left_child else 0
            right_nodes = self.right_child.count_nodes_below() if self.right_child else 0
            return 1 + left_nodes + right_nodes

    def left_child_add_prefix(self, text):
        """
        Add prefix formatting for left child in visual tree display.

        Args:
            text (str): Subtree string representation.

        Returns:
            str: Formatted string for left child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix formatting for right child in visual tree display.

        Args:
            text (str): Subtree string representation.

        Returns:
            str: Formatted string for right child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        """
        Generate a readable string representation of the tree from this node.

        Returns:
            str: A formatted tree structure string.
        """
        result = f"{'root' if self.is_root else '-> node'} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            result += self.right_child_add_prefix(self.right_child.__str__().strip())
        return result


class Leaf(Node):
    """
    Represents a leaf node in the decision tree.

    Attributes:
        value (any): The value or class label at this leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf node.

        Args:
            value (any): Output/class value for the leaf.
            depth (int): Depth of the leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
            int: Depth value.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 since this is a leaf node.

        Args:
            only_leaves (bool): Ignored.

        Returns:
            int: Always 1.
        """
        return 1


class Decision_Tree:
    """
    Main class representing the Decision Tree structure.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum samples to split a node.
        seed (int): Seed for reproducible randomness.
        split_criterion (str): Splitting strategy used.
        root (Node): Root node of the tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision_Tree object.

        Args:
            max_depth (int): Maximum tree depth.
            min_pop (int): Minimum samples for a split.
            seed (int): Random seed.
            split_criterion (str): Splitting strategy.
            root (Node): Optional root node.
        """
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
        Get the depth of the tree.

        Returns:
            int: Maximum depth.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
