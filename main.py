from typing import List

import numpy as np


##### The Tree object #####

def gini_index(t: List) -> float:
    """Two-class gini index impurity measure, i.e.: i(t) = p(0|t)p(1|t) = p(0|t)(1-p(0|t)).

    :param t: List of 2 values containing the counts for each class.
    :type t: List
    :return: Gini index of given input y.
    :rtype: float
    """
    population_size = np.sum(t)
    t1_freq = t[0] / population_size
    t2_freq = t[1] / population_size

    return t1_freq * t2_freq


class Tree:
    def __init__(self, nmin: int, minleaf: int, nfeat: int) -> None:
        """Initialisation of Tree object.

        :param nmin: Number of observations that a node must contain at least, for it to be allowed to split.
        :type nmin: int
        :param minleaf: Minimum number of observations required for a leaf node.
        :type minleaf: int
        :param nfeat: Minimal number of features considered for each
        :type nfeat: int
        """        
        # initialize the root
        self.root = None

        # stopping conditions
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat
    
    def build_tree(self, features, labels) -> Node:
        
        n_samples, n_features = features.shape

        # check the stopping criteria (nmin)
        if n_samples < self.nmin:
            # the node contains not enough observations to split so becomes a leaf Node
            leaf_value = np.bincount(labels).argmax()
            return Node(value=leaf_value)
        
        # set the number of features to be used for this choice
        features_idx = np.random.choice(np.arange(0, n_features), self.nfeat, replace=False)

        # find the best split
        parent_gain = gini_index(np.bincount(labels))
        self.find_best_split(features_idx, features, labels, parent_gain)

        # go down one branch and create the child nodes
        left_branch = self.build_tree()
        right_branch = self.build_tree()
        return Node(,, left_branch, right_branch)


    def find_best_split(self, features_idx, features, labels, p_gain):
        gains = []
        for feature_id in features_idx:
            # retrieve for each feature all unique values to get the possible thresholds
            unique_values = np.unique(features[:,feature_id])
            possible_thresholds = [np.mean(unique_values[i:i+2]) for i in range(0, len(unique_values)-2)]

            for threshold in possible_thresholds:
                # for each threshold, divide the dataset and calculate the information gain
                



class Node:
    def __init__(self, feature: int=None, threshold: float=None, left: Node=None, right: Node=None, value: str=None) -> None:
        self.feature_index = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


##### The two main functions #####
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> Tree:
    """Grows a tree that can be used to predict new cases.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param y: Vector of class labels
    :type y: np.ndarray
    :param nmin: Number of observations that a node must contain at least, for it to be allowed to split.
    :type nmin: int
    :param minleaf: Minimum number of observations required for a leaf node.
    :type minleaf: int
    :param nfeat: Minimal number of features considered for each
    :type nfeat: int
    :return: Grown and fertilized tree predicting new cases.
    :rtype: Tree
    """
    tree = Tree(nmin, minleaf, nfeat)
    tree.root = tree.build_tree(x, y)
    return tree


def tree_pred(x: np.ndarray, tr: Tree) -> np.ndarray:
    """Makes new predictions based on a grown tree.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param tr: Grown tree generated with the tree_grow function.
    :type tr: Tree
    :return: Vector of predicted class labels for the cases in x, where y[i] is the predicted class label for x[i].
    :rtype: np.ndarray
    """
    ...


##### The two auxiliary functions (for bagging and random forest) #####


if __name__ == "__main__":
    data = np.loadtxt('./data/credit.txt', delimiter=",", skiprows=1)
    
    X = data[:,:-1]
    y = data[:, -1]
    print(y)