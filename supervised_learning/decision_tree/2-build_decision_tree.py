#!/usr/bin/env python3
"""
Basic Decision Tree implementation capable of using different
split strategies, limiting tree depth, and setting a minimum
node sample size for splitting.

Contains `Node` and `Leaf` classes for representing tree structure,
and a `Decision_Tree` class to manage the construction and configuration.

Requirements:
- numpy
"""

import numpy as np


class Node:
    """
    Defines an internal node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Value used to divide the dataset.
        left_child (Node): Reference to the left branch node.
        right_child (Node): Reference to the right branch node.
        is_leaf (bool): True if the node is a terminal leaf.
        is_root (bool): True if the node is the root of the tree.
        sub_population (list): Subset of data at this node (optional).
        depth (int): Level of the node within the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Constructs a Node object.

        Args:
            feature (int): Index of the splitting feature.
            threshold (float): Value for decision split.
            left_child (Node): Node on the left after the split.
            right_child (Node): Node on the right after the split.
            is_root (bool): Flag for root node.
            depth (int): Distance from the root node.
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
        Computes the deepest level beneath the current node.

        Returns:
            int: Depth of the deepest descendant node.
        """
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Calculates the number of child nodes under this node.

        Args:
            only_leaves (bool): If True, count only the leaf nodes.

        Returns:
            int: Total count of nodes (or leaves if specified).
        """
        count = 0
        if not only_leaves:
            count += 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def right_child_add_prefix(self, text):
        """
        Helper for visual formatting of the right child node.

        Args:
            text (str): Text representation of the right subtree.

        Returns:
            str: Formatted right subtree with indentation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Helper for visual formatting of the left child node.

        Args:
            text (str): Text representation of the left subtree.

        Returns:
            str: Formatted left subtree with indentation.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Generates a readable string of the node and its subtree.

        Returns:
            str: Textual structure of the subtree from this node.
        """
        result = (
            f"{'root' if self.is_root else '-> node'} "
            f"[feature={self.feature}, threshold={self.threshold}]\n"
        )
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip()
            )
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__().strip()
            )
        return result


class Leaf(Node):
    """
    Terminal node that holds the predicted outcome.

    Attributes:
        value (any): Value or label this leaf represents.
        depth (int): Position of this node in the tree.
    """

    def __init__(self, value, depth=None):
        """
        Creates a Leaf node.

        Args:
            value (any): Prediction result for this leaf.
            depth (int): Level of the node within the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Simply returns the current depth, as it is a leaf.

        Returns:
            int: Node depth.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        A leaf is counted as one node (or one leaf).

        Args:
            only_leaves (bool): Ignored for leaf nodes.

        Returns:
            int: Always returns 1.
        """
        return 1

    def __str__(self):
        """
        Produces a string showing the leaf's value.

        Returns:
            str: Description of the leaf.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Basic classifier using a decision tree structure.

    Attributes:
        max_depth (int): Max number of levels allowed in the tree.
        min_pop (int): Minimum samples required to perform a split.
        seed (int): Seed value for reproducible randomness.
        split_criterion (str): Method used to choose splits (e.g., "random").
        root (Node): Entry point to the decision tree.
        explanatory (ndarray): Feature matrix (assigned later).
        target (ndarray): Target vector (assigned later).
        predict (callable): Prediction logic (to be implemented).
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """
        Builds a new decision tree instance.

        Args:
            max_depth (int): Limit for tree depth.
            min_pop (int): Minimum group size needed to split.
            seed (int): Random number seed.
            split_criterion (str): Rule for choosing how to split.
            root (Node): Optional custom root node.
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
        Calculates how deep the tree goes.

        Returns:
            int: Tree height from root to deepest leaf.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts total nodes or just leaves in the tree.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Converts the entire tree into a formatted string.

        Returns:
            str: Text representation of the tree structure.
        """
        return self.root.__str__()
