"""Ridge (L2-regularized) linear regression (scikit-learn)."""

from sklearn.linear_model import Ridge as SklearnRidge

from dnaffinity.models.base import BaseRegressor


class Ridge(BaseRegressor):
    def __init__(self, alpha=0.0001):
        self.model = SklearnRidge(alpha=alpha)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
