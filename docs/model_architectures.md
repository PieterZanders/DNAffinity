# Model architectures

DNAffinity ships eight regressors, all registered in `dnaffinity/models/__init__.py` and
instantiated by name via `get_model(name, **params)`. Four are scikit-learn estimators and
four are PyTorch networks. Every model implements the same interface
(`build` / `fit` / `predict` / `feature_importances` / `state_dict`) defined in
`dnaffinity/models/base.py`, so the `Model` orchestrator and the `scripts/run_*.py`
pipelines treat them interchangeably.

| Name (`--regressor`) | Backend | Input format | File |
|---|---|---|---|
| `random_forest` | scikit-learn | flat | `dnaffinity/models/random_forest.py` |
| `linear` | scikit-learn | flat | `dnaffinity/models/linear.py` |
| `ridge` | scikit-learn | flat | `dnaffinity/models/ridge.py` |
| `svr` | scikit-learn | flat | `dnaffinity/models/svr.py` |
| `DFN` | PyTorch | flat | `dnaffinity/models/dfn.py` |
| `CNN1D` | PyTorch | windows | `dnaffinity/models/cnn1d.py` |
| `CNNAttention` | PyTorch | windows | `dnaffinity/models/cnn_attention.py` |
| `Transformer` | PyTorch | windows | `dnaffinity/models/transformer.py` |

To see the concrete architecture for a given dataset, build a `Model` and print it — the
network is constructed on instantiation, before training:

```python
model = Model(dataset, len_aln, regressor='CNN1D')
print(model)   # split, input shape, trainable parameters, full module tree
```

## Class hierarchy

```
BaseRegressor                      (dnaffinity/models/base.py)
├── SklearnRegressor               wraps an estimator in self.model
│   ├── RandomForest               RandomForestRegressor(n_estimators=10)
│   ├── Linear                     LinearRegression()
│   ├── Ridge                      Ridge(alpha=0.0001)
│   └── SVR                        SVR() (RBF kernel)
└── TorchRegressor                 (dnaffinity/models/torch_base.py)
    ├── DFN                        feedforward net on flat features
    ├── CNN1DRegressor             residual 1D CNN on feature windows
    ├── CNNAttentionRegressor      CNN backbone + self-attention
    └── TransformerRegressor       Transformer encoder with CLS token
```

## Input formats

`Dataset` (`dnaffinity/dataset.py`) derives features from tetramer-level structural and
mechanical DNA properties (MD simulations). Each model declares which representation it
consumes through `input_format`:

**Flat** — `Dataset.features`, shape `(n_samples, n_features)`. All chosen feature blocks
concatenated per probe. With the standard uPBM choice
`['presence_tetramer', 'avg', 'diagonal_fce']` on 10-mers (7 tetramer windows):

| Block | Width | Content |
|---|---|---|
| `presence_tetramer` | 256 | counts of each possible tetramer (4⁴) in the probe |
| `avg` | 7 × 6 = 42 | per window: average shift, slide, rise, tilt, roll, twist |
| `diagonal_fce` | 7 × 6 = 42 | per window: diagonal of the stiffness (force-constant) matrix |
| total | 340 | |

gcPBM adds `electrostatic`; `full_fce` (full 6×6 stiffness matrix, 36 values per window),
`mgw` (minor groove width), `onehot_1mer` and `integer` encodings are also available.

**Windows** — `Dataset.windowed_features`, shape `(n_samples, seq_len, feat_dim)`. One row
per tetramer window, keeping the positional structure so convolutions/attention can act
along the sequence. Only positional feature blocks are included (`avg`, `diagonal_fce`,
`full_fce`, `mgw`, `electrostatic`); whole-probe blocks such as `presence_tetramer` are
skipped with a warning. With `avg` + `diagonal_fce` on 10-mers the tensor is
`(n, 7, 12)`.

Targets are min–max-scaled affinities; uPBM additionally supplies per-probe sample
weights used in the weighted MSE loss.

---

## scikit-learn models

Thin wrappers around standard estimators (`SklearnRegressor` provides `fit`/`predict`).
They have no epochs, so `history` is `None` and no training curve is produced.

- **`random_forest`** — `RandomForestRegressor(n_estimators=10)`. The only model exposing
  `feature_importances()`, which the pipelines turn into the feature-importance pie and
  top-features bar charts.
- **`linear`** — ordinary least squares.
- **`ridge`** — L2-regularized linear regression, `alpha=0.0001`.
- **`svr`** — support vector regression with the default RBF kernel (no sample-weight-free
  shortcuts; slowest of the four on large probe sets).

---

## PyTorch models

### Shared training machinery (`TorchRegressor`)

All four networks share one training loop in `dnaffinity/models/torch_base.py`:

- **Loss**: MSE; when sample weights are given, per-sample weighted MSE
  `mean(w · (ŷ − y)²)`.
- **Optimizer**: Adam with learning rate `lr`.
- **Batching**: sequential mini-batches of `batch_size` over `num_epochs` epochs
  (data are pre-shuffled upstream; the split itself is deterministic).
- **Progress**: tqdm bar with per-epoch train loss and, when an evaluation set is passed,
  validation loss. `Model.train()` passes the held-out test split for monitoring only.
- **History**: `regressor.history = {'train': [...], 'val': [...]}`, plotted by
  `plotting.plot_training_curve` and saved as `training_curve.png` by the scripts.
- **Device**: CUDA when available, else CPU.
- **Build**: `build(input_shape)` constructs the network from the input shape before
  training, so `print(model)` shows the full module tree up front.
- **Persistence**: pickling drops the optimizer/criterion and restores them on load;
  `state_dict()` is saved separately as `model.pth` by the scripts.

Common hyperparameters (all models): `lr`, `num_epochs`, `batch_size`,
`normalization` (input `LayerNorm` on/off). Defaults live in
`dnaffinity/common.py::DEFAULT_MODEL_PARAMS` and are mirrored by
`configs/{model}_config.json`.

### `DFN` — deep feedforward network (flat input)

A multilayer perceptron on the flat feature vector.

```
x (batch, n_features)
  → LayerNorm(n_features)                   [if normalization]
  → Linear(n_features → layers[0]) → ReLU
  → Linear(layers[i] → layers[i+1]) → ReLU  [per hidden layer]
  → Linear(... → 1)
```

| Hyperparameter | Default | Notes |
|---|---|---|
| `layers` | `[64, 32, 1]` | hidden sizes + output; last entry must be 1 |
| `lr` | `0.001` | SELEX default: `1e-4` |
| `batch_size` | `32` | SELEX default: `16` |
| `num_epochs` | `100` | |

SELEX uses a deeper variant `[256, 128, 64, 32, 1]`
(`SELEX_DFN_PARAMS` in `dnaffinity/common.py`, `configs/dfn_config.json`).
Example size: 340 input features → 24,617 trainable parameters.

### `CNN1D` — residual 1D CNN (windowed input)

Convolutions along the tetramer-window axis capture local shape/stiffness patterns.

```
x (batch, seq_len, feat_dim)
  → LayerNorm(feat_dim)                                  [if normalization]
  → transpose → Conv1d(feat_dim → hidden_dim, k) + BatchNorm + ReLU   (stem)
  → num_blocks × ResidualConvBlock:
        Conv1d(k) + BN + ReLU → Conv1d(k) + BN → +skip → ReLU
  → global pool over seq_len ('avg' or 'max')            → (batch, hidden_dim)
  → MLP head: Linear(hidden → hidden/2) → ReLU → Dropout → Linear(→ 1)
```

| Hyperparameter | Default |
|---|---|
| `hidden_dim` | 64 |
| `num_blocks` | 2 |
| `kernel_size` | 3 |
| `pool` | `'avg'` (`'max'` available) |
| `dropout` | 0.1 |

Example size: `(7, 12)` windows → 54,553 trainable parameters.
Config: `configs/cnn1d_config.json`.

### `CNNAttention` — CNN backbone + self-attention (windowed input)

Same convolutional backbone as `CNN1D`, followed by multi-head self-attention layers so
distant windows can interact before pooling.

```
x (batch, seq_len, feat_dim)
  → LayerNorm → ConvBackbone (stem + residual blocks)    → (batch, seq_len, hidden_dim)
  → num_attn_layers × [ MultiheadAttention(num_heads) → residual add → LayerNorm ]
  → global pool ('avg'/'max') → MLP head → 1
```

| Hyperparameter | Default |
|---|---|
| `hidden_dim`, `num_blocks`, `kernel_size`, `pool`, `dropout` | as CNN1D |
| `num_heads` | 4 |
| `num_attn_layers` | 1 |

The attention weights of the last layer are kept on the network
(`net.last_attention`, and `forward(x, return_attention=True)`), which allows inspecting
which windows the model attends to.
Example size: `(7, 12)` windows → 71,321 trainable parameters.
Config: `configs/cnn_attention_config.json`.

### `Transformer` — encoder with CLS token (windowed input)

A standard pre-norm Transformer encoder over the window sequence.

```
x (batch, seq_len, feat_dim)
  → LayerNorm → Linear(feat_dim → d_model)
  → + positional encoding ('sinusoidal' buffer or 'learned' embedding)
  → prepend CLS token                                     → (batch, seq_len+1, d_model)
  → num_layers × TransformerEncoderLayer(nhead, dim_feedforward, GELU, norm_first)
  → pooling: 'cls' (CLS token) or 'mean' (mean over windows)
  → MLP head → 1
```

| Hyperparameter | Default |
|---|---|
| `d_model` | 64 |
| `num_layers` | 2 |
| `nhead` | 4 |
| `dim_feedforward` | 128 |
| `dropout` | 0.1 |
| `pos_encoding` | `'sinusoidal'` (`'learned'` available) |
| `pooling` | `'cls'` (`'mean'` available) |

Example size: `(7, 12)` windows → 69,977 trainable parameters.
Config: `configs/transformer_config.json`.

---

## Choosing a model

- `random_forest` is the fast baseline and the only model with feature importances.
- `DFN` is the reference neural model on the flat physics features and usually matches or
  beats the forest on uPBM/gcPBM.
- The windowed models (`CNN1D`, `CNNAttention`, `Transformer`) only see positional
  features — they drop `presence_tetramer` — so out of the box they tend to underperform
  the flat models on short probes; they are the starting point for architectures that
  exploit the sequence dimension.

Hyperparameters can be overridden per run with `--config path/to/config.json`
(top-level keys or nested under `"pytorch_model_params"`), or in Python by passing
`pytorch_model_params={...}` to `Model`.
