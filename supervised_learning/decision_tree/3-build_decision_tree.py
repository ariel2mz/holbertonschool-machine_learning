#!/usr/bin/env python3
"""
Basic Decision Tree implementation capable of using different
split strategies, limiting tree depth, and setting a minimum
node sample size for splitting.

Contains `Node` and `Leaf` classes for representing tree structure,
and a `Decision_Tree` class to manage the construction and
configuration.

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

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
                 """ 
                 safsafsafsafsa
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
        asfsafsafsa
        """
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        safsafsafsaf
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
        safsafsafsaf
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        fsafsfafsa
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        fsafafsa
        """
        result = (f"{'root' if self.is_root else '-> node'} "
                  f"[feature={self.feature}, threshold={self.threshold}]\n")
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip())
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__().strip())
        return result

    def get_leaves_below(self):
        """
        fsafafsafsa
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf(Node):
    """
    Terminal node that holds the predicted outcome.

    Attributes:
        value (any): Value or label this leaf represents.
        depth (int): Position of this node in the tree.
    """

    def __init__(self, value, depth=None):
        """
        safafsafsaf
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        safasfsaf
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        fsafafsa
        """
        return 1

    def __str__(self):
        """
        fsafasfsa
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        safsafsa
        """
        return [self]


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

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
                 """
                 asfsafsafsa
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
        safafsafsa
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        safsafasfsa
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        asfasfsafsa
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        asfsagsafasfs
        """
        return self.root.get_leaves_below()
