"""Random forest regressor (scikit-learn)."""

from sklearn.ensemble import RandomForestRegressor

from dnaffinity.models.base import SklearnRegressor


class RandomForest(SklearnRegressor):
    def __init__(self, n_estimators=10, random_state=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def feature_importances(self):
        return self.model.feature_importances_
