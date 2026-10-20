"""Shared PyTorch training loop for all neural regressors.

Features: seeded weight init and per-epoch shuffled mini-batches, AdamW with
weight decay, optional linear warmup followed by ReduceLROnPlateau or cosine
decay, MSE or Huber loss (optionally per-sample weighted), and early stopping
on a validation set that restores the best weights.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from dnaffinity.models.base import BaseRegressor


class TorchRegressor(BaseRegressor):
    architecture = 'Torch'
    input_format = 'flat'

    def __init__(self, lr=0.001, normalization=True, num_epochs=100, batch_size=32,
                 weight_decay=1e-4, patience=15, warmup_epochs=0, scheduler='plateau',
                 loss='mse', huber_delta=1.0, seed=0, verbose=False, **model_kwargs):
        self.lr = lr
        self.normalization = normalization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        self.loss = loss
        self.huber_delta = huber_delta
        self.seed = seed
        self.verbose = verbose
        self.model_kwargs = dict(model_kwargs)
        self.model_kwargs.setdefault('normalization', normalization)
        self.net = None
        self.history = []
        self.best_epoch = None
        self._device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self._device_str)

    # -- hooks -------------------------------------------------------------
    def _check_input(self, X):
        return X

    def _build_net(self, X):
        raise NotImplementedError

    # -- training ----------------------------------------------------------
    def _loss(self, pred, target, w=None):
        if self.loss == 'huber':
            per = nn.functional.huber_loss(pred, target, reduction='none', delta=self.huber_delta)
        elif self.loss == 'mse':
            per = (pred - target) ** 2
        else:
            raise ValueError("loss must be 'mse' or 'huber'")
        if w is not None:
            return (w * per).sum() / w.sum()
        return per.mean()

    def _to_tensors(self, X, y=None, w=None):
        X = torch.tensor(self._check_input(np.asarray(X, dtype=np.float32)), device=self.device)
        y = None if y is None else torch.tensor(np.asarray(y, dtype=np.float32).ravel(), device=self.device)
        w = None if w is None else torch.tensor(np.asarray(w, dtype=np.float32).ravel(), device=self.device)
        return X, y, w

    def _lr_factor(self, epoch):
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        if self.scheduler == 'cosine':
            t = (epoch - self.warmup_epochs) / max(1, self.num_epochs - self.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return 1.0

    def fit(self, X, y, sample_weight=None, eval_set=None):
        torch.manual_seed(self.seed)
        X_t, y_t, w_t = self._to_tensors(X, y, sample_weight)
        self.net = self._build_net(X_t).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, self.patience // 3))
                   if self.scheduler == 'plateau' else None)
        if eval_set is not None:
            Xv, yv, wv = (list(eval_set) + [None])[:3]
            Xv, yv, wv = self._to_tensors(Xv, yv, wv)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float('inf'), None, 0
        self.history = []
        n = len(X_t)
        for epoch in range(self.num_epochs):
            if plateau is None or (self.warmup_epochs and epoch < self.warmup_epochs):
                for g in opt.param_groups:
                    g['lr'] = self.lr * self._lr_factor(epoch)
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:  # BatchNorm cannot train on a single sample
                    continue
                opt.zero_grad()
                pred = self.net(X_t[idx]).reshape(-1)
                loss = self._loss(pred, y_t[idx], None if w_t is None else w_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / n
            if eval_set is not None:
                self.net.eval()
                with torch.no_grad():
                    vloss = self._loss(self._predict_tensor(Xv), yv, wv).item()
                self.history.append((epoch, train_loss, vloss))
                if plateau is not None and epoch >= self.warmup_epochs:
                    plateau.step(vloss)
                if vloss < best - 1e-6:
                    best, bad, self.best_epoch = vloss, 0, epoch
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                self.history.append((epoch, train_loss, None))
            if self.verbose and (epoch % 10 == 0):
                print(f"{self.architecture} epoch {epoch + 1}: train {train_loss:.4f}"
                      + (f" val {self.history[-1][2]:.4f}" if eval_set is not None else ''))
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.epochs_trained = len(self.history)
        return self

    def _predict_tensor(self, X_t, chunk=4096):
        return torch.cat([self.net(X_t[i:i + chunk]).reshape(-1) for i in range(0, len(X_t), chunk)])

    def predict(self, X):
        X_t, _, _ = self._to_tensors(X)
        self.net.eval()
        with torch.no_grad():
            return self._predict_tensor(X_t).cpu().numpy().reshape(-1, 1)

    def state_dict(self):
        return self.net.state_dict() if self.net is not None else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('device', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.device = torch.device('cpu')
        if self.net is not None:
            self.net.to(self.device)


class WindowTorchRegressor(TorchRegressor):
    input_format = 'windows'

    def _check_input(self, X):
        if X.ndim != 3:
            raise ValueError(
                f"{self.architecture} expects X with shape (n_samples, seq_len, feat_dim), got {X.shape}")
        return X
