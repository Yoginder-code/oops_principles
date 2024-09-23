from decision_tree import DecisionTreeClassifier
from gradient_boosting import BaseGradientBoosting
import numpy as np
from metrics import Metrics 


class LightGBM(BaseGradientBoosting):

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, num_leaves=31):
        super().__init__(n_estimators, learning_rate, max_depth, random_state)
        self.num_leaves = num_leaves  # Maximum number of leaves in one tree

    def _build_tree(self, X, residuals):
        """Build a tree for LightGBM using histogram-based learning."""
        # Implement a simplified version of histogram-based learning
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, residuals)
        return tree

    def predict(self, X):
        """LightGBM-specific prediction logic."""
        return super().predict(X)

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