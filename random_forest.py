import numpy as np
from base_tree import BaseTree
from base_model import BaseModel
from decision_tree import DecisionTreeClassifier
from metrics import Metrics 

class RandomForestClassifier(BaseModel):
    
    def __init__(self, n_estimators=100, max_depth=5, random_state=None):  # Set default max_depth
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)  # Use the max_depth
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting
        return np.array([self._most_common_label(tree_preds[:, i]) for i in range(X.shape[0])])

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
    
    def _most_common_label(self, predictions):
        """Return the most common label among predictions."""
        return np.bincount(predictions).argmax()

    
