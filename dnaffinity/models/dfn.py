"""Deep feedforward network (DFN) regressor implemented in PyTorch.

Contains the network definitions (FeedForward, RegressionCV) and the DFN
wrapper that exposes the common fit/predict interface, including mini-batch
training with optional per-sample weighted MSE loss.
"""

import torch.nn as nn

from dnaffinity.models.torch_base import TorchRegressor


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


class DFN(TorchRegressor):
    """PyTorch feedforward regressor using the shared training loop."""

    architecture = 'DFN'

    def __init__(self, layers=None, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.layers = list(layers) if layers is not None else [64, 32, 1]
        self.dropout = dropout
        self.model_kwargs['layers'] = self.layers

    def _build_net(self, input_shape):
        in_features = input_shape[-1]
        net = FeedForward([in_features] + self.layers, dropout=self.dropout)
        return RegressionCV(net, in_features=in_features,
                            normalization=self.model_kwargs.get('normalization', True))
