"""Support vector regression (scikit-learn)."""

from sklearn.svm import SVR as SklearnSVR

from dnaffinity.models.base import SklearnRegressor


class SVR(SklearnRegressor):
    def __init__(self, C=1.0, epsilon=0.1, gamma='scale'):
        # expects standardized features/targets (Model scales on the training split)
        self.model = SklearnSVR(C=C, epsilon=epsilon, gamma=gamma)
