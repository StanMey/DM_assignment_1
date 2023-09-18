from collections import Counter
from typing import List

import numpy as np


##### The Tree object #####

def gini_index(t: np.ndarray) -> float:
    """Two-class gini index impurity measure, i.e.: i(t) = p(0|t)p(1|t) = p(0|t)(1-p(0|t)).

    :param t: Array containing the labels for which the gini index has to be calculated.
    :type t: np.ndarray
    :return: Gini index of given input y.
    :rtype: float
    """
    counter = np.bincount(t)
    if counter.shape[0] == 1:
        # we only have one unique label so add a 0 so the gini index gets calculated the right way
        counter = np.append(counter, [0])

    population_size = np.sum(counter)

    t1_freq = counter[0] / population_size
    t2_freq = counter[1] / population_size

    return t1_freq * t2_freq


class Tree:
    def __init__(self, nmin, minleaf, nfeat):
        # initialize the root
        self.root = None

        # stopping conditions
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat
    
    def build_tree(self, features, labels):
        
        n_samples, n_features = features.shape

        # check the stopping criteria (nmin)
        if n_samples < self.nmin:
            # the node contains not enough observations to split so becomes a leaf Node
            leaf_value = self._find_most_common(labels)
            return Node(value=leaf_value)
        
        # set the number of features to be used for this choice
        features_idx = np.random.choice(np.arange(0, n_features), self.nfeat, replace=False)

        # find the best split
        parent_gain = gini_index(labels)
        _, feature_idx, threshold = self.find_best_split(features_idx, features, labels, parent_gain)
        print(feature_idx, threshold)

        # go down one branch and create the child nodes
        
        # left_branch = self.build_tree()
        # right_branch = self.build_tree()
        # return Node(,, left_branch, right_branch)


    def find_best_split(self, features_idx, features, labels, p_gain):
        gains = []

        for feature_id in features_idx:
            # retrieve for each feature all unique values to get the possible thresholds
            unique_values = np.sort(np.unique(features[:,feature_id]))
            thresholds = [np.mean(unique_values[i:i+2]) for i in range(0, len(unique_values)-1)]

            for threshold in thresholds:
                # for each threshold, select the corresponding labels and divide the dataset
                left_idxs = np.argwhere(features[:, feature_id].flatten() <= threshold).flatten()
                right_idxs = np.argwhere(features[:, feature_id].flatten() > threshold).flatten()

                left_labels = labels[left_idxs]
                right_labels = labels[right_idxs]

                # calculate the information gain
                left_gain = gini_index(left_labels)
                right_gain = gini_index(right_labels)
                info_gain = p_gain - (left_gain * (len(left_labels) / len(labels)) + right_gain * (len(right_labels) / len(labels)))

                # save the information gain
                gains.append((info_gain, feature_id, threshold))
        
        return sorted(gains, key=lambda x: x[0], reverse=True)[0]



    def _find_most_common(self, labels):
        counter = Counter(labels)
        return counter.most_common(1)[0][0]


class Node:
    def __init__(self, feature: int=None, threshold: float=None, left=None, right=None, value: str=None):
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
    y = data[:, -1].astype(int)
    
    tree = tree_grow(X, y, 2, 1, X.shape[1])