import numpy as np
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from knn_classifier import KNearestNeighbors
from logistic_regression import LogisticRegression
from ada_boost import AdaBoostClassifier
from bagging import BaggingClassifier
from gradient_boosting import BaseGradientBoosting
from XGBoost import XGBoost  # Corrected import
from LightGBM import LightGBM
from CatBoost import CatBoost
# Create random data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, 100)  # Binary target

if __name__ == "__main__":
    # Initialize models
    xgb = RandomForestClassifier()  # Changed to XGBoost
    

    # Fit models
    xgb.fit(X, y)
    

    # Make predictions
    predictions_xgb = xgb.predict(X)
    
    # Print predictions
    print("XGBoost Predictions:", predictions_xgb)
   

    # Print performance metrics using the evaluate method
    xgb_metrics = xgb.evaluate(X, y)  # Changed to xgb

    print("XGBoost Metrics:", xgb_metrics)
