"""Support vector regression (scikit-learn)."""

from sklearn.svm import SVR as SklearnSVR

from dnaffinity.models.base import BaseRegressor


class SVR(BaseRegressor):
    def __init__(self, C=1.0, epsilon=0.1, gamma='scale'):
        # expects standardized features/targets (Model scales on the training split)
        self.model = SklearnSVR(C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)
