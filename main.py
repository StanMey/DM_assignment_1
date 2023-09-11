from typing import List

import numpy as np


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
    ...
    #


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
    def __init__(self):
        ...


class Node:
    def __init__(self, is_final: bool):
        self.is_final = is_final
