from decision_tree import DecisionTreeClassifier  # Assuming this class exists
from gradient_boosting import BaseGradientBoosting
from base_model import BaseModel
import numpy as np
from metrics import Metrics as m

class XGBoost(BaseGradientBoosting):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, reg_lambda=1):
        super().__init__(n_estimators, learning_rate, max_depth, random_state)
        self.reg_lambda = reg_lambda  # Regularization term

    def _build_tree(self, X, residuals):
        """Build a tree for XGBoost with L2 regularization."""
        # Use DecisionTreeClassifier but with a regularization term
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, residuals)
        return tree

    def predict(self, X):
        """XGBoost-specific prediction with regularization."""
        y_pred = super().predict(X)
        return y_pred - self.reg_lambda * np.sign(y_pred)  # Adding regularization

    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'accuracy': m.accuracy(y, predictions),
            'precision': m.precision(y, predictions),
            'recall': m.recall(y, predictions),
            'f1_score': m.f1_score(y, predictions),
            'confusion_matrix': m.confusion_matrix(y, predictions)
        }
        return metrics