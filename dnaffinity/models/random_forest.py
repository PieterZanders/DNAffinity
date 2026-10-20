"""Random forest regressor (scikit-learn)."""

from sklearn.ensemble import RandomForestRegressor

from dnaffinity.models.base import BaseRegressor


class RandomForest(BaseRegressor):
    def __init__(self, n_estimators=10, random_state=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
