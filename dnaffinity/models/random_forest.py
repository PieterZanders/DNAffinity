"""Random forest regressor (scikit-learn)."""

from sklearn.ensemble import RandomForestRegressor

from dnaffinity.models.base import BaseRegressor


class RandomForest(BaseRegressor):
    def __init__(self, n_estimators=200, random_state=0, n_jobs=None, min_samples_leaf=1, max_features=1.0):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,
                                           n_jobs=n_jobs, min_samples_leaf=min_samples_leaf,
                                           max_features=max_features)

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
