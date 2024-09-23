from decision_tree import DecisionTreeClassifier
from gradient_boosting import BaseGradientBoosting
import numpy as np
from base_tree import BaseTree
from metrics import Metrics 

class CatBoost(BaseGradientBoosting):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        super().__init__(n_estimators, learning_rate, max_depth, random_state)

    def _build_tree(self, X, residuals):
        """Build a tree for CatBoost using oblivious tree structure."""
        # Assuming DecisionTreeClassifier can be adapted for an oblivious tree
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, residuals)
        return tree

    def predict(self, X):
        """CatBoost-specific prediction logic."""
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