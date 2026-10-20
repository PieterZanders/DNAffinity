"""Common interface implemented by every regression model."""


class BaseRegressor:
    """Minimal fit/predict interface shared by all DNAffinity models."""

    # 'flat' models consume a 2D feature matrix (n_samples, n_features);
    # 'windows' models consume a 3D tensor (n_samples, seq_len, feat_dim)
    input_format = 'flat'

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
