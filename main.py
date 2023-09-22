from collections import Counter
from sklearn.metrics import confusion_matrix
from typing import List, Union, Tuple

import numpy as np


##### Helper functions #####
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


##### The Tree and Node classes #####
class Node:
    """The Node object containing information about either a decision or leaf.
    """
    def __init__(self, feature: int=None, threshold: float=None, left=None, right=None, value: str=None, depth:int=None, y_dist: np.ndarray=None) -> None:
        """Initialisation of Node object.

        :param feature: the id of the feature that was used for making a decision, defaults to None
        :type feature: int, optional
        :param threshold: The threshold of the decision, defaults to None
        :type threshold: float, optional
        :param left: If this Node is a decision Node this holds information about the Node connected the left side, defaults to None
        :type left: Node, optional
        :param right: If this Node is a decision Node this holds information about the Node connected the right side, defaults to None
        :type right: Node, optional
        :param value: The value of of the majority class for the leaf Node, defaults to None
        :type value: str, optional
        :param depth: The depth of the current Node, defaults to None
        :type depth: int, optional
        :param y_dist: The distribution of the y values on the leaf node (this is mainly used when printing the tree)
        :type y_dist: np.ndarray, optional
        """
        self.feature_index = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth
        self.y_dist = y_dist


def print_tree(node: Node, max_depth: int) -> str:
    """Prints the tree, beginning from the root node of the tree.

    :param node: The current node which is to be printed.
    :type node: Node
    :param max_depth: The maximal depth to which the tree is to be printed.
    :type max_depth: int
    :return: Returns a string representing the tree in textual form.
    :rtype: str
    """
    if node.depth == max_depth:
        return ""
    elif node.value is None:
        ret = "\t" * node.depth + f"feature {node.feature_index} <= {node.threshold}\n"
        ret += print_tree(node.left, max_depth)

        ret += "\t" * node.depth + f"feature {node.feature_index} > {node.threshold}\n"
        ret += print_tree(node.right, max_depth)
    else:
        ret = "\t" * node.depth + f"class: {node.value}; distr: {node.y_dist}\n"

    return ret


class Tree:
    """The Tree object containing the classification tree.
    """
    def __init__(self, nmin: int, minleaf: int, nfeat: int) -> None:
        """Initialisation of Tree object.

        :param nmin: Number of observations that a node must contain at least, for it to be allowed to split.
        :type nmin: int
        :param minleaf: Minimum number of observations required for a leaf node.
        :type minleaf: int
        :param nfeat: Minimal number of features considered for each.
        :type nfeat: int
        """
        # initialize the root
        self.root = None

        # stopping conditions
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat

    
    def build_tree(self, features: np.ndarray, labels: np.ndarray, depth: int=0) -> Node:
        """A recursive function which builds the Tree by recursively adding new decision and leaf Nodes.

        :param features: 2-dimensional data matrix containing attribute values.
        :type features: np.ndarray
        :param labels: Vector of class labels.
        :type labels: np.ndarray
        :param depth: The depth of the current Node (this is mainly used for printing the tree), defaults to 0
        :type depth: int, optional
        :return: Returns either a decision or leaf Node.
        :rtype: Node
        """
        
        n_samples, n_features = features.shape
        parent_gain = gini_index(labels)

        # check the stopping criteria (nmin) and check whether we have an optimal split
        if n_samples < self.nmin or parent_gain <= 0.0:
            # the node contains not enough observations to split or is already perfectly split, so becomes a leaf Node
            leaf_value = self._find_most_common(labels)
            return Node(value=leaf_value, depth=depth, y_dist=np.bincount(labels))
        
        # set the number of features to be used for this choice
        features_idx = np.random.choice(np.arange(0, n_features), self.nfeat, replace=False)

        # find the best split
        _, feature_idx, threshold = self.find_best_split(features_idx, features, labels, parent_gain)
        
        if feature_idx is None:
            # the minleaf constraint was violated, so make this Node a leaf node
            leaf_value = self._find_most_common(labels)
            return Node(value=leaf_value, depth=depth, y_dist=np.bincount(labels))

        # select the features for both
        left_idxs = np.argwhere(features[:, feature_idx].flatten() <= threshold).flatten()
        right_idxs = np.argwhere(features[:, feature_idx].flatten() > threshold).flatten()

        left_features, left_labels = features[left_idxs], labels[left_idxs]
        right_features, right_labels = features[right_idxs], labels[right_idxs]
        
        # go down one branch and create the child nodes
        left_branch = self.build_tree(left_features, left_labels, depth=depth + 1)
        right_branch = self.build_tree(right_features, right_labels, depth=depth + 1)
        return Node(feature_idx, threshold, left_branch, right_branch, depth=depth)


    def find_best_split(self, features_idx: np.ndarray, features: np.ndarray, labels: np.ndarray, p_gain: float) -> Union[Tuple[float, int, float], Tuple[None, None, None]]:
        """Finds the best split based on the selected features the minleaf constraint.

        :param features_idx: The id's of the features that can be considered for the current split.
        :type features_idx: np.ndarray
        :param features: 2-dimensional data matrix containing attribute values.
        :type features: np.ndarray
        :param labels: Vector of class labels.
        :type labels: np.ndarray
        :param p_gain: The impurity of the parent Node.
        :type p_gain: float
        :return: Either returns the best split found or None if the minleaf constraint couldn't be satisfied.
        :rtype: Union[Tuple[float, int, float], Tuple[None, None, None]]
        """
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
                gains.append((info_gain, feature_id, threshold, len(left_labels), len(right_labels)))
        
        # filter the gains for the minleaf requirement
        gains = [x[:3] for x in gains if x[3] >= self.minleaf and x[4] >= self.minleaf]
        sorted_gains = sorted(gains, key=lambda x: x[0], reverse=True)

        if sorted_gains:
            # there is a split that meets the minleaf constraint
            return sorted_gains[0]
        else:
            # there is no split that meets the minleaf constraint
            return None, None, None


    def _find_most_common(self, labels: np.ndarray) -> int:
        """Finds the majority label based on the labels.

        :param labels: Vector of class labels.
        :type labels: np.ndarray
        :return: Returns the majority label found in the vector.
        :rtype: int
        """
        counter = Counter(labels)
        return counter.most_common(1)[0][0]


##### The two main functions #####
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> Tree:
    """Grows a tree that can be used to predict new cases.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param y: Vector of class labels.
    :type y: np.ndarray
    :param nmin: Number of observations that a node must contain at least, for it to be allowed to split.
    :type nmin: int
    :param minleaf: Minimum number of observations required for a leaf node.
    :type minleaf: int
    :param nfeat: Minimal number of features considered for each.
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
    predictions = []

    for features in x:
        # go over every data point in the dataset
        current_node = tr.root
        print(current_node.depth)

        while current_node.value is None:
            # as long as the current node doesn't have a value, just continue down the rabbit hole
            if features[current_node.feature_index] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        # we have found a leaf node so now store the value
        predictions.append(current_node.value)
    
    return np.array(predictions)


##### The two auxiliary functions (for bagging and random forest) #####
def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> List[Tree]:
    """Grows a m trees using either the bagging or random forest approach.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param y: Vector of class labels.
    :type y: np.ndarray
    :param nmin: Number of observations that a node must contain at least, for it to be allowed to split.
    :type nmin: int
    :param minleaf: Minimum number of observations required for a leaf node.
    :type minleaf: int
    :param nfeat: Minimal number of features considered for each.
    :type nfeat: int
    :param nfeat: The number of bootstrapped samples to be drawn.
    :type nfeat: int
    :return: Grown and fertilized trees predicting new cases.
    :rtype: List[Tree]
    """
    trees = []

    for _ in range(m):
        # each iteration we draw a sample with replacement from the training set (the same size as the training set)
        new_sample = np.random.choice(np.arange(0, x.shape[0]), x.shape[0], replace=True)

        # select the correct features and labels
        sample_feat = x[new_sample]
        sample_labels = y[new_sample]

        # train a tree and save it
        trees.append(tree_grow(sample_feat, sample_labels, nmin, minleaf, nfeat))
    
    # return all trees
    return trees


def tree_pred_b(x: np.ndarray, tr: List[Tree]) -> np.ndarray:
    """Makes new predictions based on a grown trees.

    :param x: 2-dimensional data matrix containing attribute values.
    :type x: np.ndarray
    :param tr: Grown trees generated with the tree_grow_b function.
    :type tr: List[Tree]
    :return: Vector of predicted class labels for the cases in x, where y[i] is the predicted class label for x[i].
    :rtype: np.ndarray
    """
    predictions = []

    # get all predictions for each tree
    for tree in tr:

        trees_pred = tree_pred(x, tree)
        predictions.append(trees_pred)

    # select the majority vote, by getting the majority vote based on the column
    predictions = [np.argmax(np.bincount(x)) for x in np.array(predictions).T]
    return np.array(predictions)


if __name__ == "__main__":
    # check on credit dataset
    data = np.loadtxt('./data/credit.txt', delimiter=",")
    X = data[:,:-1]
    y = data[:, -1].astype(int)
    tree = tree_grow(X, y, 2, 1, X.shape[1])

    print(print_tree(tree.root, 2))

    # check on prima dataset with single tree
    # data = np.loadtxt('./data/pima.txt', delimiter=",")
    # X = data[:,:-1]
    # y = data[:, -1].astype(int)
    # tree = tree_grow(X, y, 20, 5, X.shape[1])

    # # make the predictions
    # preds = tree_pred(X, tree)
    # print(confusion_matrix(y, preds))

    # check on prima dataset with random forests
    # data = np.loadtxt('./data/pima.txt', delimiter=",")
    # X = data[:,:-1]
    # y = data[:, -1].astype(int)
    # trees = tree_grow_b(X, y, 20, 5, X.shape[1], 10)

    # # make the predictions
    # preds = tree_pred_b(X, trees)
    # print(confusion_matrix(y, preds))
