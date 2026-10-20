"""Ordinary least squares linear regression (scikit-learn)."""

from sklearn.linear_model import LinearRegression

from dnaffinity.models.base import SklearnRegressor


class Linear(SklearnRegressor):
    def __init__(self):
        self.model = LinearRegression()
