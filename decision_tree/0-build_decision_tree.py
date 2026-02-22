#!/usr/bin/env python3
"""maximum depth exercise."""


def max_depth_below(self):
    """We will take leftmost and rightmost"""
    left_depth = self.left_child.max_depth_below()
    right_depth = self.right_child.max_depth_below()
    return max(left_depth, right_depth)
