"""Deep feedforward network (DFN) regressor implemented in PyTorch.

Contains the network definitions (FeedForward, RegressionCV) and the DFN
wrapper that exposes the common fit/predict interface, including mini-batch
training with optional per-sample weighted MSE loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dnaffinity.models.base import BaseRegressor


class FeedForward(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, layers, activation='relu', dropout=0.0):
        """
        Args:
            layers: List of integers specifying the number of neurons per layer
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.dropout = dropout

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation on last layer
                if activation == 'relu':
                    modules.append(nn.ReLU())
                elif activation == 'tanh':
                    modules.append(nn.Tanh())
                elif activation == 'sigmoid':
                    modules.append(nn.Sigmoid())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


class RegressionCV(nn.Module):
    """
    Simple regression model that takes features as input and predicts affinity.
    """

    def __init__(self, model, in_features=None, normalization=True):
        """
        Args:
            model: Either a list of integers (layer sizes) or a FeedForward model
            in_features: Number of input features (required if model is a list)
            normalization: Whether to use input normalization
        """
        super().__init__()

        if isinstance(model, list):
            if in_features is None:
                raise ValueError("in_features must be provided when model is a list")
            layers = [in_features] + model
            self.nn = FeedForward(layers)
        elif isinstance(model, FeedForward):
            self.nn = model
        else:
            raise ValueError("model must be a list of integers or a FeedForward instance")

        # Input normalization (using LayerNorm instead of BatchNorm for stability)
        if normalization:
            norm_features = in_features if isinstance(model, list) else model.layers[0]
            self.norm_in = nn.LayerNorm(norm_features)
        else:
            self.norm_in = None

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """Forward pass."""
        if self.norm_in is not None:
            x = self.norm_in(x)
        return self.nn(x)


class DFN(BaseRegressor):
    """PyTorch feedforward regressor with mini-batch (optionally weighted) MSE training."""

    def __init__(self, layers=None, lr=0.001, normalization=True, num_epochs=100, batch_size=32, **_ignored):
        self.layers = layers if layers is not None else [64, 32, 1]
        self.lr = lr
        self.normalization = normalization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.net = None
        self.optimizer = None
        self.criterion = None
        self._device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self._device_str)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        in_features = X.shape[1]
        self.net = RegressionCV(model=self.layers, in_features=in_features,
                                normalization=self.normalization)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        print(f"Using simple PyTorch regression model on device: {self.device}")
        print(f"Model architecture: {in_features} -> {' -> '.join(map(str, self.layers))}")

        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        if sample_weight is not None:
            w_train = torch.tensor(np.asarray(sample_weight, dtype=np.float32),
                                   dtype=torch.float32).to(self.device)
        else:
            w_train = None

        self.net.train()
        print("Start training...")
        for epoch in range(self.num_epochs):
            total_loss = 0
            n_batches = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train))
                X_batch = X_train[i:batch_end]
                y_batch = y_train[i:batch_end]

                self.optimizer.zero_grad()
                y_pred = self.net(X_batch)

                if w_train is not None:
                    w_batch = w_train[i:batch_end]
                    loss = torch.mean(w_batch * (y_pred.squeeze() - y_batch) ** 2)
                else:
                    loss = self.criterion(y_pred.squeeze(), y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch % 20 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / n_batches:.6f}")

        print("PyTorch regression model training completed")

    def predict(self, X):
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            return self.net(X_tensor).cpu().numpy()

    def state_dict(self):
        return self.net.state_dict()

    def __getstate__(self):
        """Custom pickle state - exclude non-pickleable PyTorch objects."""
        state = self.__dict__.copy()
        state.pop('optimizer', None)
        state.pop('criterion', None)
        state.pop('device', None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling - recreate PyTorch objects."""
        self.__dict__.update(state)
        self.device = torch.device(self._device_str if torch.cuda.is_available()
                                   or self._device_str == 'cpu' else 'cpu')
        self.criterion = nn.MSELoss()
        if self.net is not None:
            self.net.to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = None
