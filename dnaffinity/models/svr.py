"""Support vector regression (scikit-learn)."""

from sklearn.svm import SVR as SklearnSVR

from dnaffinity.models.base import BaseRegressor


class SVR(BaseRegressor):
    def __init__(self):
        self.model = SklearnSVR()

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
