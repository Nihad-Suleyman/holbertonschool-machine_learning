#!/usr/bin/env python3
"""Isolation Random Tree"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Isolation Random Tree"""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Constructor"""
        self.rng = np.random.default_rng(seed)
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1
        self.explanatory = None

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

    def __str__(self):
        """String representation"""
        return self.root.__str__() + "\n"

    def depth(self):
        """Depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Update the bounds of the tree"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Get the leaves of the tree"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Update the prediction function"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0
        )

    def np_extrema(self, arr):
        """Return min and max of array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split criterion"""
        sub_population = self.explanatory[node.sub_population]
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_values = sub_population[:, feature]
            min_val, max_val = self.np_extrema(feature_values)
            diff = max_val - min_val
        x = self.rng.uniform()
        threshold = (1 - x) * min_val + x * max_val
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Return a leaf node"""
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Return a regular node"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fit nodes"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf?
        is_left_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(left_population) <= self.min_pop
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(right_population) <= self.min_pop
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit the isolation tree"""
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
