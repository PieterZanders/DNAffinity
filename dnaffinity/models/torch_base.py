"""Shared PyTorch training machinery for the neural regressors.

Subclasses only define how to build their network (_build_net) and the input
rank they expect; mini-batch training with optional per-sample weighted MSE,
tqdm progress, loss history, prediction and pickling live here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from dnaffinity.models.base import BaseRegressor

PREDICT_BATCH_SIZE = 1024


class TorchRegressor(BaseRegressor):
    architecture = None
    # number of axes of X including the sample axis (2 = flat, 3 = windows)
    input_ndim = 2

    def __init__(self, lr=0.001, normalization=True, num_epochs=100, batch_size=32,
                 **model_kwargs):
        self.lr = lr
        self.normalization = normalization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_kwargs = dict(model_kwargs)
        self.model_kwargs.setdefault('normalization', normalization)
        self.net = None
        self.input_shape = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.history = None
        self._device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self._device_str)

    def _build_net(self, input_shape):
        """Return the nn.Module for inputs of shape input_shape (no sample axis)."""
        raise NotImplementedError

    def _check_input(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != self.input_ndim:
            expected = ('(n_samples, seq_len, feat_dim)' if self.input_ndim == 3
                        else '(n_samples, n_features)')
            raise ValueError(f"{self.architecture} expects X with shape {expected}, got {X.shape}")
        return X

    def build(self, input_shape):
        """(Re)initialise the network and optimizer for the given input shape."""
        self.input_shape = tuple(input_shape)
        self.net = self._build_net(self.input_shape).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.history = None

    def _to_tensor(self, a):
        return torch.tensor(np.asarray(a, dtype=np.float32), dtype=torch.float32).to(self.device)

    def _loss(self, y_pred, y, w=None):
        y_pred = y_pred.reshape(-1)
        if w is not None:
            return torch.mean(w * (y_pred - y) ** 2)
        return self.criterion(y_pred, y)

    def _forward_batched(self, X):
        return torch.cat([self.net(X[i:i + PREDICT_BATCH_SIZE])
                          for i in range(0, len(X), PREDICT_BATCH_SIZE)])

    def fit(self, X, y, sample_weight=None, eval_set=None):
        X = self._check_input(X)
        if self.net is None or self.input_shape != X.shape[1:]:
            self.build(X.shape[1:])

        X_train, y_train = self._to_tensor(X), self._to_tensor(np.ravel(y))
        w_train = self._to_tensor(sample_weight) if sample_weight is not None else None

        val = None
        if eval_set is not None:
            X_val, y_val, *rest = eval_set
            w_val = rest[0] if rest else None
            val = (self._to_tensor(self._check_input(X_val)), self._to_tensor(np.ravel(y_val)),
                   self._to_tensor(w_val) if w_val is not None else None)

        self.history = {'train': [], 'val': [] if val is not None else None}
        n = len(X_train)
        progress = tqdm(range(self.num_epochs), desc=f'{self.architecture} [{self.device}]',
                        unit='epoch')
        for _ in progress:
            self.net.train()
            total_loss, n_batches = 0.0, 0
            for i in range(0, n, self.batch_size):
                end = min(i + self.batch_size, n)
                self.optimizer.zero_grad()
                loss = self._loss(self.net(X_train[i:end]), y_train[i:end],
                                  w_train[i:end] if w_train is not None else None)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / n_batches
            self.history['train'].append(train_loss)
            postfix = {'loss': f'{train_loss:.5f}'}
            if val is not None:
                self.net.eval()
                with torch.no_grad():
                    val_loss = self._loss(self._forward_batched(val[0]), val[1], val[2]).item()
                self.history['val'].append(val_loss)
                postfix['val_loss'] = f'{val_loss:.5f}'
            progress.set_postfix(postfix)

    def predict(self, X):
        if self.net is None:
            raise RuntimeError(f"{self.architecture} has not been trained yet; call fit() first")
        X = self._to_tensor(self._check_input(X))
        self.net.eval()
        with torch.no_grad():
            return self._forward_batched(X).cpu().numpy()

    def state_dict(self):
        return self.net.state_dict() if self.net is not None else None

    def n_parameters(self):
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def __repr__(self):
        hparams = {'lr': self.lr, 'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
                   **self.model_kwargs}
        header = f"{self.architecture}({', '.join(f'{k}={v}' for k, v in hparams.items())})"
        if self.net is None:
            return f"{header}\n  network not built yet (built from the input shape)"
        return (f"{header}\n  input shape: (batch, {', '.join(map(str, self.input_shape))})\n"
                f"  trainable parameters: {self.n_parameters():,}\n{self.net}")

    def __getstate__(self):
        """Exclude non-pickleable PyTorch objects."""
        state = self.__dict__.copy()
        state.pop('optimizer', None)
        state.pop('criterion', None)
        state.pop('device', None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling - recreate PyTorch objects."""
        state.setdefault('input_shape', None)
        state.setdefault('history', None)
        state.setdefault('model_kwargs', {})
        self.__dict__.update(state)
        self.device = torch.device(self._device_str if torch.cuda.is_available()
                                   or self._device_str == 'cpu' else 'cpu')
        self.criterion = nn.MSELoss()
        self.optimizer = None
        if self.net is not None:
            self.net.to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
