"""Model: leakage-free splitting, train-only scaling, training and evaluation.

All model-specific logic (sklearn vs PyTorch) lives in dnaffinity.models; this
class is backend-agnostic and talks to the regressors through the
BaseRegressor interface (build / fit / predict / feature_importances / state_dict).

Usage:
    model = Model(dataset, len_aln, regressor='DFN')   # builds, does not train
    print(model)                                       # split + architecture summary
    model.train()
    model.predict()

Splitting is done on Dataset.groups (one group per canonical sequence), so a
sequence, its reverse complement and any duplicates always land in the same
split. A validation split is carved out of the training groups for early
stopping. Features and targets are standardized with statistics from the
training rows only; predictions are mapped back to the original label scale
before metrics are computed.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from dnaffinity.models import PYTORCH_MODEL_NAMES, get_model


class FeatureScaler:
    """Per-feature (flat) or per-channel (windows) standardization fitted on train only."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        axes = tuple(range(X.ndim - 1))
        self.mean_ = X.mean(axis=axes)
        std = X.std(axis=axes)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)


class TargetScaler:
    """Optional log transform followed by standardization, fitted on train only."""

    def __init__(self, transform=None):
        if transform not in (None, 'log'):
            raise ValueError("target_transform must be None or 'log'")
        self.transform_name = transform

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.transform_name == 'log':
            # shift so the smallest training label maps to 1% of the training range
            self.offset_ = -y.min() + 0.01 * (y.max() - y.min() + 1e-12)
        z = self._fwd(y)
        self.mean_, self.std_ = z.mean(), (z.std() if z.std() > 1e-12 else 1.0)
        return self

    def _fwd(self, y):
        if self.transform_name == 'log':
            return np.log(np.maximum(y + self.offset_, 1e-12))
        return y

    def transform(self, y):
        return ((self._fwd(np.asarray(y, dtype=np.float64).ravel()) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, z):
        y = np.asarray(z, dtype=np.float64).ravel() * self.std_ + self.mean_
        if self.transform_name == 'log':
            return np.exp(y) - self.offset_
        return y


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    const = np.std(y_pred) < 1e-12
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        'pearson': np.nan if const else float(pearsonr(y_true, y_pred)[0]),
        'spearman': np.nan if const else float(spearmanr(y_true, y_pred)[0]),
        'r2': float(metrics.r2_score(y_true, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mae': float(metrics.mean_absolute_error(y_true, y_pred)),
    }


def group_split(groups, test_size, seed):
    """Indices (train, test) of a seeded random split on groups."""
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(idx, groups=groups))


class Model:
    def __init__(self, dataset, len_aln, regressor='random_forest', training_set_size=0.8,
                 weighted=False, pytorch_model_params=None, model_params=None, seed=0,
                 val_size=0.1, train_idx=None, test_idx=None, scale_features=True,
                 scale_target=True, target_transform=None, rc_average=False):
        """
        :param training_set_size: fraction of sequence groups used for train+val
                                  (ignored when train_idx/test_idx are given)
        :param val_size: fraction of the train+val groups held out for early stopping
                         (only used by neural models; sklearn models train on train+val)
        :param train_idx, test_idx: explicit row indices (e.g. from cross_validate)
        :param target_transform: None or 'log'
        :param rc_average: average predictions over the two strands of each sequence
        """
        self.data = dataset
        params = dict(model_params or {})
        if regressor in PYTORCH_MODEL_NAMES and pytorch_model_params:
            params.update(pytorch_model_params)
        if regressor in PYTORCH_MODEL_NAMES:
            params.setdefault('seed', seed)
        self.regressor = get_model(regressor, **params)
        self.is_torch = regressor in PYTORCH_MODEL_NAMES

        if self.regressor.input_format == 'windows':
            self.X = self.data.windowed_features
        else:
            self.X = np.asarray(self.data.features, dtype=np.float64)
        self.y = np.asarray(self.data.scores, dtype=np.float64)
        self.groups = np.asarray(getattr(self.data, 'groups', np.arange(len(self.y))))
        self.weights = np.asarray(self.data.pbm35['WEIGHT'], dtype=np.float64) if weighted else None
        self.features, self.l = [], []
        self.len_aln = len_aln
        self.training_set_size = training_set_size
        self.val_size = val_size
        self.seed = seed
        self.regressor_name = regressor
        self.scale_features, self.scale_target = scale_features, scale_target
        self.target_transform = target_transform
        self.rc_average = rc_average
        self.train_idx, self.test_idx = train_idx, test_idx
        self.y_pred = self.r2 = self.mse = self.mae = None
        self.y_train_pred = self.r2_train = None
        self.test_metrics = {}
        self.weighted = weighted
        self.trained = False
        # build the network up front so print(model) shows the full architecture
        self.regressor.build(self.X.shape[1:])

    @property
    def history(self):
        """Per-epoch losses {'train': [...], 'val': [...]} for iterative models, else None."""
        return self.regressor.history

    def __repr__(self):
        n_groups = len(np.unique(self.groups))
        if self.train_idx is not None and self.test_idx is not None:
            split = f"{len(self.train_idx)} train+val / {len(self.test_idx)} test rows"
        else:
            split = f"{self.training_set_size:.0%} of {n_groups} sequence groups for train+val (group split)"
        lines = [
            f"Model(regressor={self.regressor_name!r}, protein={self.data.protein!r})",
            f"  features: {self.regressor.input_format} {tuple(self.X.shape)}",
            f"  split:    {split}{' (weighted)' if self.weighted else ''}",
            f"  status:   {'trained' if self.trained else 'not trained'}"
            + (f", test r2 = {self.r2}, train r2 = {self.r2_train}" if self.r2 is not None else ''),
        ]
        lines += ['    ' + line for line in repr(self.regressor).splitlines()]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    def _split(self):
        if self.train_idx is None or self.test_idx is None:
            self.train_idx, self.test_idx = group_split(self.groups, 1 - self.training_set_size, self.seed)
        trv = np.asarray(self.train_idx)
        if self.is_torch and self.val_size:
            tr, va = group_split(self.groups[trv], self.val_size, self.seed + 1)
            self.fit_idx, self.val_idx = trv[tr], trv[va]
        else:
            self.fit_idx, self.val_idx = trv, np.array([], dtype=int)
        assert not set(self.groups[self.fit_idx]) & set(self.groups[self.test_idx])

    def _prep_X(self, idx):
        X = self.X[idx]
        return self.feature_scaler.transform(X) if self.scale_features else np.asarray(X, dtype=np.float32)

    def train(self):
        """Split on sequence groups, fit train-only scalers and train the regressor."""
        y = self.y.ravel()
        self._split()
        self.feature_scaler = FeatureScaler().fit(self.X[self.fit_idx]) if self.scale_features else None
        self.target_scaler = (TargetScaler(self.target_transform).fit(y[self.fit_idx])
                              if (self.scale_target or self.target_transform) else None)
        tf = (self.target_scaler.transform if self.target_scaler is not None
              else (lambda v: np.asarray(v, dtype=np.float32)))

        self.X_train, self.y_train = self._prep_X(self.fit_idx), y[self.fit_idx]
        self.X_test, self.y_test = self._prep_X(self.test_idx), y[self.test_idx]
        self.w_train = None if self.weights is None else self.weights[self.fit_idx]
        self.w_test = None if self.weights is None else self.weights[self.test_idx]
        kwargs = {'sample_weight': self.w_train}
        if self.is_torch and len(self.val_idx):
            wv = None if self.weights is None else self.weights[self.val_idx]
            kwargs['eval_set'] = (self._prep_X(self.val_idx), tf(y[self.val_idx]), wv)
        self.regressor.fit(self.X_train, tf(self.y_train), **kwargs)
        self.features, self.l = [], []
        self._extract_feature_importances()
        self.trained = True
        return self

    def _raw_predict(self, X, idx):
        z = np.asarray(self.regressor.predict(X), dtype=np.float64).ravel()
        if self.rc_average and idx is not None:
            # mean prediction over all rows (both strands) of the same sequence group
            g = self.groups[idx]
            z = pd.Series(z).groupby(g).transform('mean').values
        return self.target_scaler.inverse_transform(z) if self.target_scaler is not None else z

    def _group_level(self, y, p, idx):
        """Collapse rows to one value per sequence group (strands share the label)."""
        df = pd.DataFrame({'g': self.groups[idx], 'y': np.ravel(y), 'p': np.ravel(p)})
        agg = df.groupby('g').mean()
        return agg['y'].values, agg['p'].values

    def predict(self, testing_dataset=None):
        """
        Predict the internal test split (or an external Dataset) and set metrics:
        r2, mse (true mean squared error), mae, rmse, pearson, spearman. With
        rc_average the internal test metrics are computed per unique sequence
        (strand-averaged prediction), otherwise per strand (row).
        """
        if not self.trained:
            raise RuntimeError("Model is not trained yet; call model.train() first")
        if testing_dataset is not None:
            Xe = (testing_dataset.windowed_features if self.regressor.input_format == 'windows'
                  else testing_dataset.features)
            Xe = self.feature_scaler.transform(Xe) if self.scale_features else Xe
            self.y_test = np.asarray(testing_dataset.scores, dtype=np.float64).ravel()
            z = np.asarray(self.regressor.predict(Xe), dtype=np.float64).ravel()
            self.y_pred = (self.target_scaler.inverse_transform(z) if self.target_scaler is not None else z)
            yt, yp = self.y_test, self.y_pred
        else:
            self.y_pred = self._raw_predict(self.X_test, self.test_idx)
            if self.rc_average:
                # one (strand-averaged) prediction per unique sequence
                yt, yp = self._group_level(self.y_test, self.y_pred, self.test_idx)
            else:
                # every strand is scored separately
                yt, yp = self.y_test, self.y_pred
        self.y_pred = self.y_pred.reshape(-1, 1)
        self.y_test = np.asarray(self.y_test).reshape(-1, 1)
        self.test_metrics = regression_metrics(yt, yp)
        self.r2 = np.round(self.test_metrics['r2'], 3)
        self.mse = self.test_metrics['mse']
        self.mae = self.test_metrics['mae']

        self.y_train_pred = self._raw_predict(self.X_train, self.fit_idx).reshape(-1, 1)
        self.r2_train = np.round(metrics.r2_score(self.y_train, self.y_train_pred.ravel()), 3)
        return self.test_metrics

    # ------------------------------------------------------------------
    def _extract_feature_importances(self):
        """Classify each feature by type and record its importance (tree models only)."""
        importances = self.regressor.feature_importances()
        if importances is None:
            return
        indices = np.argsort(importances)[::-1]
        num_presence = 256
        num_avg_per_pos = 6
        num_diag_per_pos = 6
        avg_names = ['AVG Shift', 'AVG Slide', 'AVG Rise', 'AVG Tilt', 'AVG Roll', 'AVG Twist']
        diag_names = ['Diagonal Shift', 'Diagonal Slide', 'Diagonal Rise',
                      'Diagonal Tilt', 'Diagonal Roll', 'Diagonal Twist']
        for f in range(self.X.shape[1]):
            if indices[f] < num_presence:
                feat = 'Presence'
            elif indices[f] < num_presence + (self.len_aln - 3) * num_avg_per_pos:
                feat = avg_names[(indices[f] - num_presence) % num_avg_per_pos]
            elif indices[f] < num_presence + (self.len_aln - 3) * (num_avg_per_pos + num_diag_per_pos):
                pos_idx = (indices[f] - num_presence - (self.len_aln - 3) * num_avg_per_pos) % num_diag_per_pos
                feat = diag_names[pos_idx]
            else:
                feat = 'Electro'
            self.l.append(importances[indices[f]])
            self.features.append([feat, importances[indices[f]]])


def cross_validate(dataset, len_aln, regressor, n_folds=5, seed=0, verbose=True, **model_kwargs):
    """
    Shuffled GroupKFold cross-validation on sequence groups.
    :return: DataFrame with one row of test metrics per fold (+ train r2, epochs)
    """
    groups = np.asarray(dataset.groups)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for k, (tr, te) in enumerate(gkf.split(np.arange(len(groups)), groups=groups)):
        m = Model(dataset, len_aln, regressor, train_idx=tr, test_idx=te, seed=seed + k, **model_kwargs).train()
        res = dict(fold=k, **m.predict(), r2_train=float(m.r2_train),
                   epochs=getattr(m.regressor, 'epochs_trained', None),
                   n_train=len(m.fit_idx), n_val=len(m.val_idx), n_test_groups=len(np.unique(groups[te])))
        if verbose:
            print(f"[{regressor}] fold {k}: r={res['pearson']:.3f} rho={res['spearman']:.3f} r2={res['r2']:.3f}")
        rows.append(res)
    return pd.DataFrame(rows)
