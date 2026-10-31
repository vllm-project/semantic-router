# complete code
"""
RADAR-inspired dataset for difficulty and model-budget ability estimates.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class RadarDataset:
    def __init__(self, n_samples=1000, n_features=10):
        self.n_samples = n_samples
        self.n_features = n_features
        self.X, self.y = self.generate_data()

    def generate_data(self):
        # Generate classification data
        X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_informative=5, n_redundant=3, n_repeated=2, n_classes=2)
        # Add difficulty and model-budget ability features
        X_difficulty = np.random.rand(self.n_samples, 1)
        X_model_budget = np.random.rand(self.n_samples, 1)
        X = np.hstack((X, X_difficulty, X_model_budget))
        return X, y

    def train_test_split(self):
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)