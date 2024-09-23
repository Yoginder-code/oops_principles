import numpy as np
from base_model import BaseModel

class KNearestNeighbors(BaseModel):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single_point(x) for x in X]
        return np.array(predictions)

    def _predict_single_point(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()

    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'accuracy': self.accuracy(y, predictions),
            'precision': self.precision(y, predictions),
            'recall': self.recall(y, predictions),
            'f1_score': self.f1_score(y, predictions),
            'confusion_matrix': self.confusion_matrix(y, predictions)
        }
        return metrics
