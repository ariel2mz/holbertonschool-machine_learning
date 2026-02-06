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
        """
        Initialize an internal node in the decision tree.

        Args:
            feature: Index of feature used for splitting
            threshold: Threshold value for splitting
            left_child: Left child node (values <= threshold)
            right_child: Right child node (values > threshold)
            is_root: Boolean indicating if this is the root node
            depth: Depth of this node in the tree
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
        self.indicator = None

    def max_depth_below(self):
        """
        Calculate the maximum depth of the tree below this node.

        Returns:
            Maximum depth below this node
        """
        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves: If True, count only leaf nodes

        Returns:
            Number of nodes below this node
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
        Format the right child subtree string with proper prefix.

        Args:
            text: String representation of right child subtree

        Returns:
            Formatted string with right child prefix
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def left_child_add_prefix(self, text):
        """
        Format the left child subtree string with proper prefix.

        Args:
            text: String representation of left child subtree

        Returns:
            Formatted string with left child prefix
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def __str__(self):
        """
        Generate string representation of the node and its subtree.

        Returns:
            String representation of the node
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
        Get all leaf nodes below this node.

        Returns:
            List of leaf nodes below this node
        """
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Update the lower and upper bounds for all nodes below this node.
        Propagates bounds recursively through the tree.
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

    def update_indicator(self):
        """
        Update the indicator function for this node.
        Creates a lambda function that checks if input data points
        fall within the node's bounds.
        """
        def is_large_enough(A):
            """
            Check if data points are above all lower bounds.

            Args:
                A: Input data array

            Returns:
                Boolean array indicating which points satisfy lower bounds
            """
            return np.all(np.array([A[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(A):
            """
            Check if data points are below or equal to all upper bounds.

            Args:
                A: Input data array

            Returns:
                Boolean array indicating which points satisfy upper bounds
            """
            return np.all(np.array([A[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda A: np.all(np.array([is_large_enough(A),
                                                    is_small_enough(A)]),
                                          axis=0)

    def pred(self, x):
        """
        Predict the value for a single data point using tree traversal.

        Args:
            x: Single data point

        Returns:
            Predicted value
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Terminal node that holds the predicted outcome.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf node.

        Args:
            value: Predicted value for this leaf
            depth: Depth of this leaf in the tree
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
        Get the depth of this leaf node.

        Returns:
            Depth of this leaf
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this leaf (always 1).

        Args:
            only_leaves: Ignored for leaf nodes

        Returns:
            Always returns 1
        """
        return 1

    def __str__(self):
        """
        Generate string representation of the leaf.

        Returns:
            String representation of the leaf
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Get leaves below this node (just this leaf).

        Returns:
            List containing this leaf
        """
        return [self]

    def update_bounds_below(self):
        """
        Update bounds for leaf node (no operation needed).
        """
        pass

    def update_indicator(self):
        """
        Update the indicator function for this leaf node.
        Creates a lambda function that checks if input data points
        fall within the leaf's bounds.
        """
        def is_large_enough(A):
            """
            Check if data points are above all lower bounds.

            Args:
                A: Input data array

            Returns:
                Boolean array indicating which points satisfy lower bounds
            """
            return np.all(np.array([A[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(A):
            """
            Check if data points are below or equal to all upper bounds.

            Args:
                A: Input data array

            Returns:
                Boolean array indicating which points satisfy upper bounds
            """
            return np.all(np.array([A[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda A: np.all(np.array([is_large_enough(A),
                                                    is_small_enough(A)]),
                                          axis=0)

    def pred(self, x):
        """
        Predict the value for a single data point (leaf node).

        Args:
            x: Single data point (ignored for leaf)

        Returns:
            The leaf's value
        """
        return self.value

class Decision_Tree:
    """
    asfasfsafsafas
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
                 """
                 fasfsafsafasfas
                 """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def np_extrema(self, arr):
        """
        afsfsafsafas
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        afssfsafasfsafsa
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_vals = self.explanatory[:, feature][node.sub_population]
            f_min, f_max = self.np_extrema(f_vals)
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        safsafsafsa
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(target, dtype=bool)
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {self.accuracy(explanatory,target)}""")

    def fit_node(self, node):
        """
        asfasfasfas
        """
        if (np.sum(node.sub_population) < self.min_pop or
            node.depth >= self.max_depth or
            np.all(self.target[node.sub_population] ==
                   self.target[node.sub_population][0])):
            node.value = np.bincount(
                self.target[node.sub_population]).argmax()
            node.is_leaf = True
            return

        node.feature, node.threshold = self.split_criterion(node)
        X_sub = self.explanatory[node.sub_population, node.feature]
        left_population = node.sub_population.copy()
        left_population[node.sub_population] = X_sub > node.threshold
        right_population = node.sub_population.copy()
        right_population[node.sub_population] = X_sub <= node.threshold

        is_left_leaf = (np.sum(left_population) <= self.min_pop or
                        np.all(self.target[left_population] ==
                               self.target[left_population][0]) or
                        node.depth + 1 >= self.max_depth)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (np.sum(right_population) <= self.min_pop or
                         np.all(self.target[right_population] ==
                                self.target[right_population][0]) or
                         node.depth + 1 >= self.max_depth)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        agfsgasfsafsafsa
        """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        asgasgasgagasfas
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        afsgagsfsafsa
        """
        return np.sum(self.predict(test_explanatory) == test_target) / test_target.size

