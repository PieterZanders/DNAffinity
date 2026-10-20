"""Support vector regression (scikit-learn)."""

from sklearn.svm import SVR as SklearnSVR

from dnaffinity.models.base import SklearnRegressor


class SVR(SklearnRegressor):
    def __init__(self):
        self.model = SklearnSVR()
