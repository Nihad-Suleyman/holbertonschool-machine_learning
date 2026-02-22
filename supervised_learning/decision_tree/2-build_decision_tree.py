#!/usr/bin/env python3
"""another decision tree."""


class Decision_Tree:
    """Represents a decision tree."""

    def __str__(self):
        """Return the string representation of the tree."""
        return self.root.__str__()


class Leaf:
    """Represents a leaf node."""

    def __str__(self):
        """Return the string representation of a leaf."""
        return f"-> leaf [value={self.value}]"


class Node:
    """Represents an internal node."""

    def left_child_add_prefix(self, text):
        """Add left-branch prefix formatting."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add right-branch prefix formatting."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return the string representation of the subtree."""
        head = "root" if self.is_root else "-> node"
        out = f"{head} [feature={self.feature}, threshold={self.threshold}]\n"
        out += self.left_child_add_prefix(str(self.left_child))
        out += self.right_child_add_prefix(str(self.right_child))
        return out.rstrip("\n")
