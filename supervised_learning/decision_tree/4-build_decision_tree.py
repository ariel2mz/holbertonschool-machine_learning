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
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize an internal node with optional children and settings.
        """
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

    def max_depth_below(self):
        """
        Return the maximum depth beneath this node.
        """
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes or leaves below this node.

        Args:
            only_leaves (bool): Count only leaves if True.

        Returns:
            int: Number of nodes or leaves below.
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
        Format right child subtree for printing.

        Args:
            text (str): Subtree string.

        Returns:
            str: Formatted subtree with right-child prefix.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Format left child subtree for printing.

        Args:
            text (str): Subtree string.

        Returns:
            str: Formatted subtree with left-child prefix.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Return a formatted string of this node and its subtree.

        Returns:
            str: Node and subtree description.
        """
        if self.is_leaf:
            return f"-> leaf [value={self.value}]"
        
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
        Return a list of all leaves under this node.

        Returns:
            list: List of Leaf instances.
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively update lower and upper bounds of each node below.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()
                
                if child == self.left_child:
                    child.lower[self.feature] = min(
                        child.upper.get(self.feature, np.inf), 
                        self.threshold)
                else:
                    child.upper[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), 
                        self.threshold)
                
                child.update_bounds_below()


class Leaf(Node):
    """
    Terminal node that holds the predicted outcome.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf node with a value and optional depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """
        Return the depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Return count for this leaf (always 1).

        Args:
            only_leaves (bool): Ignored.

        Returns:
            int: Always 1.
        """
        return 1

    def __str__(self):
        """
        Return a string representing this leaf.

        Returns:
            str: Leaf description with its value.
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Return this leaf in a single-element list.

        Returns:
            list: [self]
        """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder for leaf bounds update (does nothing).
        """
        pass


class Decision_Tree:
    """
    Basic classifier using a decision tree structure.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a decision tree with optional root node.
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
        Return the maximum depth of the tree.

        Returns:
            int: Maximum depth from root to leaves.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes or leaves in the tree.

        Args:
            only_leaves (bool): Count only leaves if True.

        Returns:
            int: Number of nodes or leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Return a string representation of the entire tree.

        Returns:
            str: Tree as formatted string.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Return all leaves of the tree.

        Returns:
            list: List of Leaf instances.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update lower and upper bounds for all nodes in the tree.
        """
        self.root.update_bounds_below()
