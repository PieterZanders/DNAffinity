"""Ridge (L2-regularized) linear regression (scikit-learn)."""

from sklearn.linear_model import Ridge as SklearnRidge

from dnaffinity.models.base import SklearnRegressor


class Ridge(SklearnRegressor):
    def __init__(self, alpha=1.0):
        self.model = SklearnRidge(alpha=alpha)
