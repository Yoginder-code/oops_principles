import numpy as np

class BaseModel:
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None,
                 regularization=None, class_weight=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.class_weight = class_weight

    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method!")

    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method!")

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

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    def confusion_matrix(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])

    def initialize_weights(self, n_features):
        if self.random_state:
            np.random.seed(self.random_state)
        return np.random.randn(n_features)

    def update_weights(self, weights, gradients):
        return weights - self.learning_rate * gradients
