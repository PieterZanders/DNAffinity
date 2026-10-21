"""Common interface implemented by every regression model."""


class BaseRegressor:
    """Minimal build/fit/predict interface shared by all DNAffinity models."""

    # 'flat' models consume a 2D feature matrix (n_samples, n_features);
    # 'windows' models consume a 3D tensor (n_samples, seq_len, feat_dim)
    input_format = 'flat'

    # per-epoch losses ({'train': [...], 'val': [...]}); None for non-iterative models
    history = None

    def build(self, input_shape):
        """Prepare the model for inputs of shape input_shape (excluding the sample axis)."""

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Train on X, y (optionally sample-weighted). eval_set=(X_val, y_val[, w_val]) is
        used by neural models for early stopping and ignored by sklearn models."""
        raise NotImplementedError

    def predict(self, X):
        """Return predictions for features X as a numpy array."""
        raise NotImplementedError

    def feature_importances(self):
        """Per-feature importances, or None if the model does not provide them."""
        return None

    def state_dict(self):
        """PyTorch state dict for neural models, None for sklearn models."""
        return None


class SklearnRegressor(BaseRegressor):
    """Thin wrapper around a scikit-learn estimator stored in self.model."""

    model = None

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def __repr__(self):
        return repr(self.model)
