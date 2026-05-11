'''
This file was created by copying the code from logistic_regression_breast_cancer.ipynb.
In that notebook, the algorithm was implemented from scratch to facilitate a clearer
comparison between the code and the underlying mathematical formulas. 
'''

import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
z = np.linspace(-8, 8, 400)
gz = sigmoid(z)

def log_likelihood(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    h = sigmoid(X @ theta)
    eps = 1e-15
    h = np.clip(h, eps, 1- eps)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000, lambda_reg: float = 0.01) -> None:
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.lambda_reg = lambda_reg
        self.theta = None
        self.history = []
        self.val_acc_history = []
    
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val:np.ndarray | None = None,
            checkpoint: int = 50
    ) -> "LogisticRegressionScratch":
        m, n = X.shape
        self.theta = np.zeros(n)

        penalty_mask = np.ones(n)
        penalty_mask[0] = 0.0

        for i in range(self.n_iter):
            h = sigmoid(X @ self.theta)
            gradient = X.T @ (y - h)
            gradient -= self.lambda_reg * self.theta * penalty_mask
            self.theta += self.lr * gradient

            self.history.append(log_likelihood(X, y, self.theta))

            if X_val is not None and i % checkpoint == 0:
                self.val_acc_history.append(accuracy_score(y_val, self.predict(X_val)))
                
        return self
        

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        "Return P(y=1|x)"
        return sigmoid(X @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionScratch":
        instance = joblib.load(path)
        print(f"Loaded from {path}")
        return instance