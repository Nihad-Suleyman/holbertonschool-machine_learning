#!/usr/bin/env python3
"""Decision tree classes."""


class Leaf:
    """Represents a leaf node holding a prediction value."""

    def __init__(self, value, depth=0):
        """Initialize a leaf with its value and depth."""
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.is_root = False

    def count_nodes_below(self, only_leaves=False):
        """Return 1 since a leaf is always a single node."""
        return 1

    def __str__(self):
        """Return the printable representation of a leaf."""
        return f"-> leaf [value={self.value}]"


class Node:
    """Represents an internal node that splits on a feature and threshold."""

    def __init__(self, feature, threshold, left_child, right_child, depth=0, is_root=False):
        """Initialize a node with split info, children, and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """Recursively count nodes or leaves in the subtree."""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        return left + right if only_leaves else 1 + left + right

    def left_child_add_prefix(self, text):
        """Add left-branch prefix formatting."""
        lines = text.split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "    |  " + line + "\n"
        return out

    def right_child_add_prefix(self, text):
        """Add right-branch prefix formatting."""
        lines = text.split("\n")
        out = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            out += "       " + line + "\n"
        return out

    def __str__(self):
        """Return the printable representation of the subtree."""
        label = "root" if self.is_root else "-> node"
        out = f"{label} [feature={self.feature}, threshold={self.threshold}]\n"
        out += self.left_child_add_prefix(str(self.left_child))
        out += self.right_child_add_prefix(str(self.right_child))
        return out.rstrip("\n")


class Decision_Tree:
    """Represents a decision tree with a root node."""

    def __init__(self, root):
        """Initialize the tree with its root."""
        self.root = root

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes or leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return the printable representation of the tree."""
        return self.root.__str__()