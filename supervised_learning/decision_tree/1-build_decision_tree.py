#!/usr/bin/env python3
"""Decision Tree structure with node counting functionality."""


class Node:
    """Represents an internal decision tree node."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 depth=0, is_root=False):
        """Initialize an internal node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """Recursively count nodes or leaves in the subtree."""
        left = self.left_child.count_nodes_below(only_leaves)
        right = self.right_child.count_nodes_below(only_leaves)

        if only_leaves:
            return left + right
        return 1 + left + right


class Leaf:
    """Represents a leaf node in the tree."""

    def __init__(self, value, depth=0):
        """Initialize a leaf node."""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """Return 1 since a leaf counts as one node."""
        return 1


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root):
        """Initialize the tree with a root node."""
        self.root = root

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves)
