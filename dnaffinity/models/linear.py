"""Ordinary least squares linear regression (scikit-learn)."""

from sklearn.linear_model import LinearRegression

from dnaffinity.models.base import BaseRegressor


class Linear(BaseRegressor):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
