import numpy as np
from base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None):
        super().__init__(learning_rate, max_iter, random_state)
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train logistic regression using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = self.initialize_weights(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights = self.update_weights(self.weights, dw)
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict binary labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)
