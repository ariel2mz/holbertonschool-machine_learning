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
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}
        self.indicator = None

    def max_depth_below(self):

        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        safsafsafsafsa
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
        safsafsafsafsa
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        safsafsafsafsa
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        safsafsafsafsa
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
        safsafsafsafsa
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        safsafsafsafsa
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.upper[self.feature] = self.threshold
                elif child == self.right_child:
                    child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        safsafsafsafsa
        """
        def is_large_enough(A):
        """
        safsafsafsafsa
        """
            return np.all(np.array([A[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(A):
        """
        safsafsafsafsa
        """
            return np.all(np.array([A[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda A: np.all(np.array([is_large_enough(A),
                                                    is_small_enough(A)]),
                                          axis=0)


class Leaf(Node):
    """
    Terminal node that holds the predicted outcome.
    """

    def __init__(self, value, depth=None):
        """
        safsafsafsafsa
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.lower = {}
        self.upper = {}
        self.indicator = None

    def max_depth_below(self):
        """
        safsafsafsafsa
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        safsafsafsafsa
        """
        return 1

    def __str__(self):
        """
        safsafsafsafsa
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        safsafsafsafsa
        """
        return [self]

    def update_bounds_below(self):
        """
        safsafsafsafsa
        """
        pass

    def update_indicator(self):
        """
        safsafsafsafsa
        """
        def is_large_enough(A):
        """
        safsafsafsafsa
        """
            return np.all(np.array([A[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(A):
        """
        safsafsafsafsa
        """
            return np.all(np.array([A[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda A: np.all(np.array([is_large_enough(A),
                                                    is_small_enough(A)]),
                                          axis=0)


class Decision_Tree:
    """
    Basic classifier using a decision tree structure.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        safsafsafsafsa
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
        safsafsafsafsa
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        safsafsafsafsa
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        safsafsafsafsa
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        safsafsafsafsa
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        safsafsafsafsa
        """
        self.root.update_bounds_below()
