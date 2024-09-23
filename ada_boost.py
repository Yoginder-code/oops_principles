import numpy as np
from base_model import BaseModel
from decision_tree import DecisionTreeClassifier

class AdaBoostClassifier(BaseModel):
    def __init__(self, n_estimators=50, learning_rate=1.0):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # Initialize weights

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y)
            predictions = model.predict(X)

            # Calculate error
            miss = [int(x) for x in (predictions != y)]  # Indicator function
            err = np.dot(w, miss)  # Weighted error rate

            # Calculate alpha
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / (err + 1e-10))
            self.alphas.append(alpha)

            # Update weights
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)  # Normalize to sum to 1

            self.models.append(model)

    def predict(self, X):
        final_pred = np.sum([alpha * model.predict(X) for model, alpha in zip(self.models, self.alphas)], axis=0)
        return np.sign(final_pred)

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
