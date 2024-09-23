import numpy as np
from base_tree import BaseTree
from metrics import Metrics 

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier(BaseTree):
    def __init__(self, max_depth=100, random_state=None):
        super().__init__(max_depth, random_state)
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples <= 2):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        best_feature, best_thresh = self._best_split(X, y, n_features)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return DecisionTreeNode(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature in range(n_features):
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold

        return split_idx, split_thresh

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'accuracy': Metrics.accuracy(y, predictions),
            'precision': Metrics.precision(y, predictions),
            'recall': Metrics.recall(y, predictions),
            'f1_score': Metrics.f1_score(y, predictions),
            'confusion_matrix': Metrics.confusion_matrix(y, predictions)
        }
        return metrics
