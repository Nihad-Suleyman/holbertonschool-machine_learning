#!/usr/bin/env python3
""" 0-build_decision_tree.py """
import numpy as np


class Node:
    """ Node """
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        """ Initialize the node """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Maximum depth below the current Node """
        max_d = self.depth
        if self.left_child:
            max_d = max(max_d, self.left_child.max_depth_below())
        if self.right_child:
            max_d = max(max_d, self.right_child.max_depth_below())
        return max_d

    def count_nodes_below(self, only_leaves=False):
        """ Count nodes """
        c = int(not only_leaves)
        if self.left_child:
            c += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child:
            c += self.right_child.count_nodes_below(only_leaves=only_leaves)
        return c

    def left_child_add_prefix(self, text):
        """ left """
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "| " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """ right """
        lines = text.split("\n")
        new_text = "+---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += x + "\n"
        return new_text

    def __str__(self):
        """ __str__ """
        n = "root" if self.is_root else "node"
        s = f"{n} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            s += self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            s += self.right_child_add_prefix(
                self.right_child.__str__().strip()
            )
        return s


class Leaf(Node):
    """ Leaf """
    def __init__(self, value, depth=None):
        """ Initialize the leaf """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Max depth below the leaf (its own depth) """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Count nodes below the leaf (always 1 for a leaf) """
        return 1

    def __str__(self):
        """ String representation of the leaf """
        return (f"leaf [value={self.value}]")


class Decision_Tree():
    """ Decision Tree """
    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        """ Initialize a tree """
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

    def depth(self):
        """ Depth of the tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count nodes in the tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ String representation of the tree """
        return self.root.__str__()