from typing import List

import numpy as np


##### The two main functions #####
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int):
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
    sample_size, feature_size = np.shape(x)

    if sample_size >= nmin:
        split = best_split(x, y, minleaf)
        if split["quality"] < 0:
            left_tree = tree_grow(split["left"], split["left_y"], nmin, minleaf, nfeat)
            right_tree = tree_grow(split["left"], split["left_y"], nmin, minleaf, nfeat)
            return Node(feature=split["index"], threshold=split["value"],
                        left_node=left_tree, right_node=right_tree, impurity=split["impurity"])

    y = list(y)
    leaf_value = max(y, key=y.count)
    return Node(value=leaf_value)


def tree_pred(x: np.ndarray, tr) -> np.ndarray:
    """Makes new predictions based on a grown tree.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param tr: Grown tree generated with the tree_grow function.
    :type tr: Tree
    :return: Vector of predicted class labels for the cases in x, where y[i] is the predicted class label for x[i].
    :rtype: np.ndarray
    """
    ...


def best_split(x: np.ndarray, y: np.ndarray, minleaf: int) -> dict:
    split = {}
    best_gain = 1
    x = np.column_stack((x, y.T))
    for index in range(x.shape[1]):
        features = x[:, index]
        features = np.unique(features)

        for threshold in features:
            data_left = x[x[:, 0] > threshold]
            data_right = x[x[:, 0] <= threshold]

            # print(data_left.shape[0], data_right.shape[0])
            if data_left.shape[0] >= minleaf and data_right.shape[0] >= minleaf:
                y, y_left, y_right = x[:, -1], data_left[:, -1], data_right[:, -1]

                info_gain = quality_of_split(y, y_left, y_right)
                # print("index, thres, impurity",index, threshold, info_gain)

                print(info_gain)
                if info_gain < best_gain:
                    #print("dat: ", x, index, threshold, "data left:", data_left,"data right", data_right)
                    split["index"] = index
                    split["value"] = threshold
                    split["quality"] = info_gain
                    split["impurity"] = gini_index(y)
                    split["left"] = data_left
                    split["right"] = data_right
                    split["left_y"] = y_left
                    split["right_y"] = y_right
                    best_gain = info_gain
    return split


def gini_index(y: np.ndarray) -> float:
    """Two-class gini index impurity measure, i.e.: i(t) = p(0|t)p(1|t) = p(0|t)(1-p(0|t)).

    :param t: List of 2 values containing the counts for each class.
    :type t: List
    :return: Gini index of given input y.
    :rtype: float
    """
    _, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)

    if len(counts) == 1:
        return 0

    return counts[0] * counts[1]


def quality_of_split(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray):
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    print("weights", weight_left, gini_index(y_left), weight_right, gini_index(y_right))

    return gini_index(y) - (weight_left * gini_index(y_left) + weight_right * gini_index(y_right))


class Tree:
    def __init__(self):
        self.root = None

    def run(self, x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int):
        self.root = tree_grow(x, y, nmin, minleaf, nfeat)

    def print_tree(self, tree=None):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X" + str(tree.feature), "<=", tree.threshold, "!", tree.impurity)
            print("left ")
            self.print_tree(tree.left_node)
            print("right")
            self.print_tree(tree.right_node)


class Node:
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, impurity=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.impurity = impurity
        self.value = value


data = [[22, 0, 0, 28, 1, 0],
        [46, 0, 1, 32, 0, 0],
        [24, 1, 1, 24, 1, 1],
        [25, 0, 0, 27, 1, 1]]
# [29,1,1,32,0,0],
# [45,1,1,30,0,1],
# [63,1,1,58,1,1],
# [36,1,0,52,1,1],
# [23,0,1,40,0,1],
# [50,1,1,28,0,1]]
data = np.asarray(data)

y = data[:, -1]
data = np.delete(data, np.s_[-1:], axis=1)
print(data, y)

tree = Tree()
tree.run(data, y, 2, 1, 5)
tree.print_tree()
