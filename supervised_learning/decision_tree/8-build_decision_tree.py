#!/usr/bin/env python3
"""A module to build a tree"""
import numpy as np


class Node:
    """A class to represent a node in a decision tree"""

    def __init__(self,
                 feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False, depth=0):
        """Constructor method"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the decision tree"""
        if self.is_leaf:
            return self.depth
        else:
            return max(
                self.left_child.max_depth_below(),
                self.right_child.max_depth_below()
            )

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)
        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def __str__(self):
        """Return a string representation of the decision tree"""
        if self.is_root:
            prefix = "root"
        else:
            prefix = "-> node"

        result = (f"{prefix} [feature={self.feature}, "
                  f"threshold={self.threshold}]\n")
        result += self.left_child_add_prefix(self.left_child.__str__())
        result += self.right_child_add_prefix(self.right_child.__str__())
        return result

    def left_child_add_prefix(self, text):
        """Add a left child prefix to the decision tree"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Add a right child prefix to the decision tree"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += ("       " + x) + "\n"
        return (new_text)

    def get_leaves_below(self):
        """Return leaves below the decision tree"""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update the bounds below the decision tree"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Update the indicator function"""
        def is_large_enough(x):
            # Check all features against lower bounds
            return np.all(
                np.array(
                    [np.greater(x[:, key],
                                self.lower[key])
                     for key in self.lower.keys()]),
                axis=0)

        def is_small_enough(x):
            # Check all features against upper bounds
            return np.all(
                np.array(
                    [np.less_equal(x[:, key],
                                   self.upper[key])
                     for key in self.upper.keys()]),
                axis=0)

        # Combine element-wise (logical AND)
        self.indicator = lambda x: is_large_enough(x) & is_small_enough(x)

    def pred(self, x):
        """Prediction"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """A class to represent a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the decision tree"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        return 1

    def __str__(self):
        """Return a string representation of the decision tree"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Return leaves below the decision tree"""
        return [self]

    def update_bounds_below(self):
        """Update the bounds below the decision tree"""
        pass

    def pred(self, x):
        """Prediction"""
        return self.value


class Decision_Tree():
    """A class to represent a decision tree"""

    def __init__(self,
                 max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None
                 ):
        """Constructor method"""
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
        """Return the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Return leaves below the decision tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds below the decision tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Predict the decision tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0
        )

    def pred(self, x):
        """Prediction"""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
                self.explanatory, self.target)}""")

    def np_extrema(self, arr):
        """Extrema"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split criterion"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(
                0,
                self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature]
                [node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """Fitting the node"""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
                self.explanatory[:, node.feature] > node.threshold
        )

        right_population = node.sub_population & (
                self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf ?
        is_left_leaf = (
                node.depth + 1 >= self.max_depth or
                np.sum(left_population) <= self.min_pop or
                np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
                node.depth + 1 >= self.max_depth or
                np.sum(right_population) <= self.min_pop or
                np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Getting leaf child"""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Getting node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Accuracy"""
        return np.sum(np.equal(
            self.predict(test_explanatory),
            test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """Possible thresholds"""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """GIni split criterion one feature"""
        n = node.sub_population
        t = self.explanatory[n, feature]
        c = self.target[n]

        thresholds = self.possible_thresholds(node, feature)

        if thresholds.size == 0:
            return 0, np.inf

        classes = np.unique(c)
        n_classes = classes.size

        gini_scores = []

        for threshold in thresholds:

            left_mask = t > threshold
            right_mask = t <= threshold

            left_y = c[left_mask]
            right_y = c[right_mask]

            def gini(labels):
                """Function for labels"""
                if labels.size == 0:
                    return 0
                probs = np.bincount(labels) / labels.size
                return 1 - np.sum(probs ** 2)

            gini_left = gini(left_y)
            gini_right = gini(right_y)

            weight_left = left_y.size / c.size
            weight_right = right_y.size / c.size

            gini_avg = weight_left * gini_left + weight_right * gini_right

            gini_scores.append(gini_avg)

        gini_scores = np.array(gini_scores)

        best_index = np.argmin(gini_scores)

        return thresholds[best_index], gini_scores[best_index]

    def Gini_split_criterion(self, node):
        """Gini split criterion"""
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
