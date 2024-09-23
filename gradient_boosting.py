# gradient_boosting.py
import numpy as np
from base_tree import BaseTree
from decision_tree import DecisionTreeClassifier
from metrics import Metrics 

class BaseGradientBoosting(BaseTree):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        super().__init__(max_depth, random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        y_pred = np.zeros(y.shape)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return np.round(y_pred)

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

    # Ensure you have the metric methods defined as well (accuracy, precision, etc.)
