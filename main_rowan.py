from typing import List
from sklearn.metrics import accuracy_score, confusion_matrix
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
    instances, n_features = np.shape(x)

    if instances >= nmin:
        feature_idx = np.random.choice(np.arange(0, n_features), nfeat, replace=False)
        split = best_split(x, y, minleaf, feature_idx)
        if split["quality"] > 0:
            left_tree = tree_grow(split["left"], split["left_y"], nmin, minleaf, nfeat)
            right_tree = tree_grow(split["right"], split["right_y"], nmin, minleaf, nfeat)
            return Node(feature=split["index"], threshold=split["value"],
                        left_node=left_tree, right_node=right_tree, impurity=split["impurity"], info_gain=split["quality"])

    y = list(y)
    if (y.count(0) > y.count(1)):
        return Node(value= 0)
    else: return Node(value= 1)

def tree_pred(x: np.ndarray, tr) -> np.ndarray:
    """Makes new predictions based on a grown tree.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param tr: Grown tree generated with the tree_grow function.
    :type tr: Tree
    :return: Vector of predicted class labels for the cases in x, where y[i] is the predicted class label for x[i].
    :rtype: np.ndarray
    """
    pred_list = []
    if x.ndim == 1:
        pred_list.append(predict(x, tr))
    else:
        for instance in x:
            pred_list.append(predict(instance, tr))
    return np.asarray(pred_list)

def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int):
    tree_list = []
    x_y = np.column_stack((x, y.T))
    for i in range(m):
        tree = Tree()
        sampled_indices = np.random.choice(x.shape[0], x.shape[0], replace=True)
        new_x = x_y[sampled_indices, :]
        new_x = new_x[:,:-1]
        new_y = x_y[:, -1]

        tree.run(new_x, new_y, nmin, minleaf, nfeat)
        tree_list.append(tree)
    return tree_list

def tree_pred_b(x: np.ndarray, tree_lst: list):
    all_preds=[]
    for instance in x:
        preds_forrest = []
        for tree in tree_lst:
            preds_forrest.append(tree_pred(instance, tree.root))
        if (preds_forrest.count(0) > preds_forrest.count(1)):
            all_preds.append(0)
        else: all_preds.append(1)
    return np.asarray(all_preds)
   
def predict(instance: np.ndarray, tr) -> float:

    if tr.value != None: return tr.value
    value = instance[tr.feature]
    if value > tr.threshold:
        return predict(instance, tr.left_node)
    else: return predict(instance, tr.right_node)

def best_split(x: np.ndarray, y: np.ndarray, minleaf: int, feature_idx: np.ndarray) -> dict:
    split = {}
    best_gain = -1
    x = np.column_stack((x, y.T))

    for index in feature_idx:
        features = x[:, index]
        features = np.unique(features)
        features = [np.mean(features[i:i+2]) for i in range(0, len(features)-1)]

        for threshold in features:

            data_left = x[x[:, index] > threshold]
            data_right = x[x[:, index] <= threshold]

            if data_left.shape[0] >= minleaf and data_right.shape[0] >= minleaf:

                y, y_left, y_right = x[:, -1], data_left[:, -1], data_right[:, -1]
                info_gain = quality_of_split(y, y_left, y_right)
                data_left = data_left[:,:-1]
                data_right = data_right[:,:-1]

                if info_gain > best_gain:
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

def quality_of_split(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray):
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    gain = gini_index(y) - (weight_left * gini_index(y_left) + weight_right * gini_index(y_right))
    return gain

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
            print("feature " + str(tree.feature), " | threshold: <=", tree.threshold, "| gini value: ", tree.impurity)
            print("left node:")
            self.print_tree(tree.left_node)
            print("right node:")
            self.print_tree(tree.right_node)



class Node:
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, impurity=None, info_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.impurity = impurity
        self.info_gain = info_gain
        self.value = value


def read_data(text_file: str):
    data = []
    with open(text_file, 'r') as file:
        for line in file:

            row = line.strip().split(',')
            row = [int(row[0])] + [float(elem) for elem in row[1:]]
            data.append(row)

    return np.asarray(data)

data = read_data('data/pima.txt')
#data = read_data('data/credit.txt')


x = np.delete(data, np.s_[-1:], axis=1)
y = data[:, -1]

def run_algo(forrest=True):
    if(forrest):
        # create tree object
        tree = Tree()

        # run it
        tree.run(x, y, 20, 5, 8)
        tree.print_tree()

        #make predictions
        preds = tree_pred(data, tree.root)

        print(confusion_matrix(y, preds), accuracy_score(y, preds))
    else:
        lst = tree_grow_b(x,y,20,5,8, 27)
        preds = tree_pred_b(x, lst)
        print(confusion_matrix(y, preds), accuracy_score(y, preds))

run_algo(forrest=False)
