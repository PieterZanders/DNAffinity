import torch
import torch.nn as nn

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
        
        # Build network
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
    Similar to mlcolvar RegressionCV but simplified.
    """
    def __init__(self, model, in_features=None, normalization=True):
        """
        Args:
            model: Either a list of integers (layer sizes) or a FeedForward model
            in_features: Number of input features (required if model is a list)
            normalization: Whether to use input normalization
        """
        super().__init__()
        
        # Handle model input
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
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        """Forward pass."""
        if self.norm_in is not None:
            x = self.norm_in(x)
        return self.nn(x)
    
    def forward_cv(self, x):
        """Alias for forward (for compatibility)."""
        return self.forward(x)
    
    def training_step(self, batch, weights=None):
        """
        Compute training loss.
        
        Args:
            batch: Dict with 'data' (features) and 'target' (labels), or tuple (x, y)
            weights: Optional sample weights
        """
        # Handle different input formats
        if isinstance(batch, dict):
            x = batch['data']
            labels = batch['target']
            if 'weights' in batch:
                weights = batch['weights']
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, labels = batch
        else:
            raise ValueError("batch must be a dict with 'data' and 'target' or a tuple (x, y)")
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss
        if weights is not None:
            # Weighted MSE loss
            loss = torch.mean(weights * (y_pred.squeeze() - labels.squeeze()) ** 2)
        else:
            loss = self.loss_fn(y_pred.squeeze(), labels.squeeze())
        
        return loss

