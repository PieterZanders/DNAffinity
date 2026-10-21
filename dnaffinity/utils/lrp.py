"""Layer-wise relevance propagation (epsilon-LRP) for the DFN and CNN1D models.

Walks a trained network backwards and redistributes each prediction onto the
input features with the epsilon rule (Bach et al. 2015):

- DFN:   relevance per flat feature,   shape (n_samples, n_features)
- CNN1D: relevance per window feature, shape (n_samples, seq_len, feat_dim)

Propagation rules used here:
- Linear / Conv1d / BatchNorm (eval): epsilon rule via the autograd trick
- ReLU / Dropout: relevance passes through unchanged
- residual connections: relevance splits proportionally to each branch
- global pooling: 'avg' distributes over windows, 'max' goes to the argmax
- the input LayerNorm is treated as a pass-through, so relevance aligns
  one-to-one with the original features

Layer biases absorb part of the relevance, so per-sample relevance sums only
approximately equal the prediction (exactly, for bias-free networks).
"""

import numpy as np
import torch
import torch.nn as nn


def _stabilize(z, eps):
    return z + eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))


def _eps_rule(layer_fn, a, R, eps):
    """Relevance at the input of an affine layer, given relevance R at its output."""
    a = a.detach().requires_grad_(True)
    z = _stabilize(layer_fn(a), eps)
    s = (R / z).detach()
    (c,) = torch.autograd.grad((z * s).sum(), a)
    return (a * c).detach()


def _lrp_dfn(net, x, eps):
    """Propagate through RegressionCV: LayerNorm -> [Linear -> ReLU]* -> Linear."""
    a = net.norm_in(x) if net.norm_in is not None else x
    layers = list(net.nn.network)

    activations, h = [], a
    with torch.no_grad():
        for layer in layers:
            activations.append(h)
            h = layer(h)

    R = h  # start from the prediction
    for layer, a_in in zip(reversed(layers), reversed(activations)):
        if isinstance(layer, nn.Linear):
            R = _eps_rule(layer, a_in, R, eps)
        # ReLU / Dropout: pass-through
    return R  # (B, n_features); LayerNorm treated as pass-through


def _lrp_cnn1d(net, x, eps):
    """Propagate through CNN1D: LayerNorm -> conv stem -> residual blocks -> pool -> head."""
    h = net.norm_in(x) if net.norm_in is not None else x
    t = h.transpose(1, 2)  # (B, feat_dim, seq_len)

    conv_s, bn_s, _relu = net.backbone.stem

    def stem_affine(a):
        return bn_s(conv_s(a))

    with torch.no_grad():
        stem_out = torch.relu(stem_affine(t))

        block_ins, branch_outs = [], []
        b = stem_out
        for blk in net.backbone.blocks:
            block_ins.append(b)
            branch = blk.bn2(blk.conv2(torch.relu(blk.bn1(blk.conv1(b)))))
            branch_outs.append(branch)
            b = torch.relu(branch + b)

        seq = b.transpose(1, 2)  # (B, seq_len, hidden)
        if net.pool == 'avg':
            def pool_fn(s):
                return s.mean(dim=1)
        else:
            def pool_fn(s):
                return s.max(dim=1).values
        pooled = pool_fn(seq)

        head_layers = list(net.head.net)
        head_acts, hh = [], pooled
        for layer in head_layers:
            head_acts.append(hh)
            hh = layer(hh)

    # ---- backwards ----
    R = hh
    for layer, a_in in zip(reversed(head_layers), reversed(head_acts)):
        if isinstance(layer, nn.Linear):
            R = _eps_rule(layer, a_in, R, eps)

    R = _eps_rule(pool_fn, seq, R, eps)  # relevance per window, (B, seq_len, hidden)
    R = R.transpose(1, 2)  # (B, hidden, seq_len)

    for blk, b_in, branch in zip(reversed(list(net.backbone.blocks)),
                                 reversed(block_ins), reversed(branch_outs)):
        # outer ReLU: pass-through; sum node: split proportionally to contributions
        denom = _stabilize(branch + b_in, eps)
        R_branch = R * branch / denom
        R_skip = R * b_in / denom
        with torch.no_grad():
            inner = torch.relu(blk.bn1(blk.conv1(b_in)))
        R_branch = _eps_rule(lambda a, blk=blk: blk.bn2(blk.conv2(a)), inner, R_branch, eps)
        R_branch = _eps_rule(lambda a, blk=blk: blk.bn1(blk.conv1(a)), b_in, R_branch, eps)
        R = R_branch + R_skip

    R = _eps_rule(stem_affine, t, R, eps)
    return R.transpose(1, 2)  # (B, seq_len, feat_dim); LayerNorm pass-through


_PROPAGATORS = {'DFN': _lrp_dfn, 'CNN1D': _lrp_cnn1d}


def lrp_relevance(model, X=None, eps=1e-6, batch_size=512):
    """
    Epsilon-LRP relevance of the input features for a trained DFN or CNN1D.

    :param model: trained dnaffinity Model, or the regressor itself
    :param X: inputs to explain; defaults to the Model's test split
    :param eps: stabilizer of the epsilon rule
    :param batch_size: samples per backward pass
    :return: np.ndarray, (n, n_features) for DFN or (n, seq_len, feat_dim) for CNN1D
    """
    regressor = getattr(model, 'regressor', model)
    name = getattr(regressor, 'architecture', None)
    if name not in _PROPAGATORS:
        raise ValueError(
            f"LRP is implemented for {sorted(_PROPAGATORS)}, got "
            f"{name or type(regressor).__name__!r}"
        )
    if regressor.net is None:
        raise RuntimeError("The regressor has not been trained yet; call train()/fit() first")
    if X is None:
        X = getattr(model, 'X_test', None)
        if X is None:
            raise ValueError("No X given and the model has no test split to default to")

    propagate = _PROPAGATORS[name]
    net = regressor.net
    net.eval()

    X = np.asarray(X, dtype=np.float32)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(regressor.device)
        out.append(propagate(net, xb, eps).cpu().numpy())
    return np.concatenate(out)
