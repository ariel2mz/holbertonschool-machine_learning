#!/usr/bin/env python3
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, 
                left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Doc """
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
        """ Doc """
        if self.is_leaf:
            return 1
        if only_leaves:
            if self.left_child:
                left_leaves = self.left_child.count_nodes_below(True)
            else:
                left_leaves = 0
            if self.right_child:
                right_leaves = self.right_child.count_nodes_below(True)
            else:
                right_leaves = 0
            return right_leaves + left_leaves
        else:
            if self.left_child:
                left_leaves = self.left_child.count_nodes_below()
            else:
                left_leaves = 0
            if self.right_child:
                right_leaves = self.right_child.count_nodes_below()
            else:
                right_leaves = 0
            return right_leaves + left_leaves + 1

    def left_child_add_prefix(self,text):
        lines=text.split("\n")
        new_text="    +--"+lines[0]+"\n"
        for x in lines[1:] :
            new_text+=("    |  "+x)+"\n"
        return (new_text)    

    def right_child_add_prefix(self,text):
        lines=text.split("\n")
        new_text="    +--"+lines[0]+"\n"
        for x in lines[1:] :
            new_text+=("       "+x)+"\n"
        return (new_text)

    def __str__(self):
        """ This method returns a string representation of the node and its
        subtree for easy viewing. """
        result = f"{'root' if self.is_root else '-> node'} \
[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__().strip())


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        return self.depth

    def count_nodes_below(self, only_leaves=False) :
        return 1


class Decision_Tree():
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self) :
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False) :
        return self.root.count_nodes_below(only_leaves=only_leaves)
