"""Model: train/test splitting, training orchestration and evaluation.

All model-specific logic (sklearn vs PyTorch) lives in dnaffinity.models; this
class is backend-agnostic and talks to the regressors through the
BaseRegressor interface (build / fit / predict / feature_importances / state_dict).
Plotting helpers live in dnaffinity.plotting.

Usage:
    model = Model(dataset, len_aln, regressor='DFN')   # builds, does not train
    print(model)                                       # architecture summary
    model.train()
    model.predict()
"""

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from dnaffinity.models import PYTORCH_MODEL_NAMES, get_model


class Model:
    def __init__(self, dataset, len_aln, regressor='random_forest', training_set_size=0.8,
                 weighted=False, pytorch_model_params=None):
        self.data = dataset

        params = dict(pytorch_model_params) if (regressor in PYTORCH_MODEL_NAMES
                                                and pytorch_model_params) else {}
        self.regressor = get_model(regressor, **params)

        # each model declares the input it consumes: a flat feature matrix or
        # per-window feature tensors
        if self.regressor.input_format == 'windows':
            self.X = self.data.windowed_features
        else:
            self.X = self.data.features
        self.y = self.data.scores
        if weighted:
            self.weights = self.data.pbm35['WEIGHT']
        else:
            self.weights = None
        self.features = []
        self.l = []
        self.len_aln = len_aln
        self.training_set_size = training_set_size
        self.regressor_name = regressor
        self.y_pred, self.r2, self.mse = 3 * [None]
        self.y_train_pred, self.r2_train = 2 * [None]
        self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test = 6 * [None]
        self.weighted = weighted
        self.trained = False

        # build the network up front so print(model) shows the full architecture
        self.regressor.build(self.X.shape[1:])

    @property
    def history(self):
        """Per-epoch losses {'train': [...], 'val': [...]} for iterative models, else None."""
        return self.regressor.history

    def __repr__(self):
        n_train = int(len(self.y) * self.training_set_size)
        lines = [
            f"Model(regressor={self.regressor_name!r}, protein={self.data.protein!r})",
            f"  features: {self.regressor.input_format} {tuple(self.X.shape)}",
            f"  split:    {n_train} train / {len(self.y) - n_train} test"
            f"{' (weighted)' if self.weighted else ''}",
            f"  status:   {'trained' if self.trained else 'not trained'}"
            + (f", test r2 = {self.r2}, train r2 = {self.r2_train}" if self.r2 is not None else ''),
            "  architecture:",
        ]
        lines += ['    ' + line for line in repr(self.regressor).splitlines()]
        return '\n'.join(lines)

    def train(self):
        """
        Splits the data from the Dataset and trains the regressor, also extracting data
        on feature importances. The test split is passed as an evaluation set so that
        iterative models record a validation loss curve (it is not used for fitting).
        :return: self
        """
        y = self.y.ravel() if len(self.y.shape) > 1 else self.y

        # splits w/o randomness for training and testing
        if self.weights is not None:
            self.X_train, self.X_test, self.y_train, self.y_test, \
                self.w_train, self.w_test = train_test_split(self.X, y, self.weights,
                                                             train_size=self.training_set_size,
                                                             random_state=0, shuffle=False)
        else:
            self.X_train, self.X_test, self.y_train, \
                self.y_test = train_test_split(self.X, y, train_size=self.training_set_size,
                                               random_state=0, shuffle=False)
            self.w_train = None
            self.w_test = None

        self.regressor.fit(self.X_train, self.y_train, sample_weight=self.w_train,
                           eval_set=(self.X_test, self.y_test, self.w_test))

        self.features, self.l = [], []
        self._extract_feature_importances()
        self.trained = True
        return self

    def _extract_feature_importances(self):
        """Classify each feature by type and record its importance (tree models only)."""
        importances = self.regressor.feature_importances()
        if importances is None:
            return
        indices = np.argsort(importances)[::-1]

        # Determine feature types based on feature indices. This assumes the feature
        # structure: presence_tetramer (256), avg (6 per position), diagonal_fce (6 per
        # position), electrostatic
        num_presence = 256  # 4^4 = 256 possible tetramers
        num_avg_per_pos = 6
        num_diag_per_pos = 6
        avg_names = ['AVG Shift', 'AVG Slide', 'AVG Rise', 'AVG Tilt', 'AVG Roll', 'AVG Twist']
        diag_names = ['Diagonal Shift', 'Diagonal Slide', 'Diagonal Rise',
                      'Diagonal Tilt', 'Diagonal Roll', 'Diagonal Twist']

        for f in range(self.X.shape[1]):
            if indices[f] < num_presence:
                feat = 'Presence'
            elif indices[f] < num_presence + (self.len_aln - 3) * num_avg_per_pos:
                pos_idx = (indices[f] - num_presence) % num_avg_per_pos
                feat = avg_names[pos_idx]
            elif indices[f] < num_presence + (self.len_aln - 3) * (num_avg_per_pos + num_diag_per_pos):
                pos_idx = (indices[f] - num_presence - (self.len_aln - 3) * num_avg_per_pos) % num_diag_per_pos
                feat = diag_names[pos_idx]
            else:
                feat = 'Electro'

            self.l.append(importances[indices[f]])
            self.features.append([feat, importances[indices[f]]])

    def predict(self, testing_dataset=None):
        """
        Performs the prediction on the test data from the internal (default) or external dataset;
        sets the predictions and corresponding errors/correlations as attributes
        :param testing_dataset: Dataset instance, test data to supply for cross-evaluation (optional)
        :return: None
        """
        if not self.trained:
            raise RuntimeError("Model is not trained yet; call model.train() first")
        if testing_dataset is not None:
            self.X_test = (testing_dataset.windowed_features
                           if self.regressor.input_format == 'windows'
                           else testing_dataset.features)
            self.y_test = testing_dataset.scores
        self.y_pred = np.asarray(self.regressor.predict(self.X_test)).reshape(-1, 1)
        self.y_test = np.asarray(self.y_test).reshape(-1, 1)
        self.r2 = np.round(metrics.r2_score(self.y_test, self.y_pred), 3)
        self.mse = np.round(metrics.mean_absolute_error(self.y_test, self.y_pred), 6)

        # predictions on the training split, to compare against the test performance
        if self.X_train is not None:
            self.y_train_pred = np.asarray(self.regressor.predict(self.X_train)).reshape(-1, 1)
            self.r2_train = np.round(metrics.r2_score(np.asarray(self.y_train).reshape(-1, 1),
                                                      self.y_train_pred), 3)
