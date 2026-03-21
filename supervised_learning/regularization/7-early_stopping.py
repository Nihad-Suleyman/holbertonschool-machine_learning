#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count

    return False, count
