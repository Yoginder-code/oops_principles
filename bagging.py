import numpy as np
from base_model import BaseModel
from decision_tree import DecisionTreeClassifier
from metrics import Metrics

class BaggingClassifier(BaseModel):
    def __init__(self, n_estimators=10, base_estimator=DecisionTreeClassifier):
        super().__init__()
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self._bootstrap(X, y)
            model = self.base_estimator()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.array([self._most_common_label(pred) for pred in predictions.T])

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, predictions):
        """
        Return the most common label from predictions.
        """
        unique, counts = np.unique(predictions, return_counts=True)
        return unique[np.argmax(counts)]

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
