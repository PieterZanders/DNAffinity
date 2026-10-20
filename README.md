# DNAffinity

#### Machine Learning method to predict binding sites for different Transcription Factors using data from different experimental techniques.

DNAffinity presents a physics-based machine learning approach to predict *in vitro* 
transcription factor binding affinities from structural and mechanical DNA properties 
directly derived from atomistic molecular dynamics simulations. The method is able to 
predict affinities obtained with high-throughput techniques as different as uPBM, gcPBM 
and HT-SELEX. When complemented with chromatin structure information, our *in vitro* 
trained method provides also good estimates of *in vivo* binding sites in yeast.

## Project Layout

---------------

```
dnaffinity/            # the Python package
├── dataset.py         # Dataset: sequence/affinity tables -> feature matrices
├── model.py           # Model: train/test split, evaluation
├── common.py          # shared CLI, config loading, training and saving helpers
├── plotting.py        # prediction scatter and feature-importance plots
└── models/            # one file per regression model
    ├── base.py        # common fit/predict interface
    ├── random_forest.py
    ├── linear.py
    ├── ridge.py
    ├── svr.py
    ├── dfn.py         # PyTorch feedforward network (flat features)
    ├── cnn1d.py       # PyTorch 1D-CNN (per-window features)
    ├── cnn_attention.py  # PyTorch 1D-CNN + self-attention (per-window features)
    └── transformer.py # PyTorch Transformer encoder (per-window features)

scripts/               # command-line entry points (one per experimental technique)
├── run_upbm.py
├── run_selex.py
└── run_gcpbm.py

legacy/                # original research analyses, kept for reference
├── chromosome.py      # in vivo yeast analysis and figures
└── gcpbm_final_yeast.py  # Colab notebook export (not runnable as-is)

data/
├── input/             # tetramer property tables (avg, FCE, MGW, electrostatics)
├── upbm/Gata4/        # uPBM test case
├── selex/Gata4/       # HT-SELEX test case
└── gcpbm/cbf1/        # gcPBM test case (plus yeast chromosome files)

configs/               # example JSON configs for the DFN model
notebooks/             # preprocessing notebooks and R pipeline
outputs/               # created at runtime (trained models, intermediate files)
```

## Installation

---------------

Required dependencies:
- Python 3.x
- NumPy, Pandas, Matplotlib
- scikit-learn
- PyTorch (for `DFN`)

Optionally install the package (from the repository root):

```bash
pip install -e .
```

The scripts also run directly from the repository without installation.

## Usage

---------------

The pipeline for a specific protein consists of:

1. Preprocessing the data (specific steps for each experimental technique; details on upbm and selex can be found under the **notebooks** folder)
2. Getting the preprocessed data file (e.g. the `Gata4_training.txt` file from **data/upbm/Gata4**)
3. Running the corresponding regressor with command-line arguments

### Required Files

Make sure you have all the required files (compare formatting with the templates). All
scripts read from `{data_dir}/{protein}/`:
- `{data_dir}/{protein}/{protein}_training.txt` for `run_upbm.py`
- `{data_dir}/{protein}/{cycle}.txt` for `run_selex.py`
- `{data_dir}/{protein}/{protein}_{concentration}.txt` and `{data_dir}/{protein}/{protein}_freq_matrix_6.txt` for `run_gcpbm.py`

### Command-Line Interface

All regressor scripts support command-line arguments for flexible configuration:

#### uPBM Regressor

```bash
python scripts/run_upbm.py --protein PROTEIN [OPTIONS]
```

**Arguments:**
- `--protein` (required): Protein name

**Options:**
- `--data-dir PATH`: Data directory (default: `data/upbm`)
- `--regressor TYPE`: Regressor type - `random_forest`, `linear`, `ridge`, `svr`, `DFN` (default: `DFN`)
- `--training-set-size FLOAT`: Training set size fraction (default: `0.9`)
- `--randomize-fce`: Randomize FCE features
- `--score STRING`: Score column name (default: `Median_intensity`)
- `--model-target TYPE`: Model target - `octamers` or `tetramers` (default: `octamers`)
- `--config PATH`: Path to JSON file with PyTorch model parameters

**Example:**
```bash
python scripts/run_upbm.py --protein Gata4 --regressor DFN --config configs/dfn_config.json
```

#### SELEX Regressor

```bash
python scripts/run_selex.py --protein PROTEIN --cycle CYCLE [OPTIONS]
```

**Arguments:**
- `--protein` (required): Protein name
- `--cycle` (required): Cycle number

**Options:** Same as uPBM regressor, except `--data-dir` defaults to `data/selex`

**Example:**
```bash
python scripts/run_selex.py --protein Gata4 --cycle 4 --regressor DFN --training-set-size 0.9
```

#### gcPBM Regressor

```bash
python scripts/run_gcpbm.py --protein PROTEIN --concentration CONCENTRATION [OPTIONS]
```

**Arguments:**
- `--protein` (required): Protein name
- `--concentration` (required): Concentration value

**Options:** Same as uPBM regressor, except `--data-dir` defaults to `data/gcpbm`

**Example:**
```bash
python scripts/run_gcpbm.py --protein cbf1 --concentration 100 --regressor DFN --config configs/dfn_config.json
```

### Regressor Types

Each model lives in its own module under `dnaffinity/models/`:

- **`random_forest`**: Random Forest regressor (scikit-learn)
- **`linear`**: Linear Regression (scikit-learn)
- **`ridge`**: Ridge Regression (scikit-learn)
- **`svr`**: Support Vector Regression (scikit-learn)
- **`DFN`**: PyTorch feedforward neural network (recommended)
- **`CNN1D`**: PyTorch residual 1D-CNN over per-window features
- **`CNNAttention`**: PyTorch 1D-CNN with self-attention over per-window features
- **`Transformer`**: PyTorch Transformer encoder over per-window features

The sklearn models and the DFN consume the flat feature matrix
(`Dataset.features`); CNN1D, CNNAttention and Transformer declare
`input_format = 'windows'` and consume the per-window tensor
(`Dataset.windowed_features`, shape `(n_samples, seq_len, feat_dim)`).

To add a new model, create a module implementing the `BaseRegressor` interface
(`fit`, `predict`, optionally `feature_importances`) and register it in
`dnaffinity/models/__init__.py`.

### PyTorch Model Configuration

Every PyTorch model has its own config file in `configs/`, one per model:
`dfn_config.json`, `cnn1d_config.json`, `cnn_attention_config.json`,
`transformer_config.json`. Pass the matching file via `--config` (or none to
use the built-in defaults, which mirror these files):

```bash
python scripts/run_upbm.py --protein Gata4 --regressor Transformer --config configs/transformer_config.json
```

**Example (`configs/dfn_config.json`):**
```json
{
  "layers": [256, 128, 64, 32, 1],
  "lr": 0.0005,
  "normalization": true,
  "num_epochs": 150,
  "batch_size": 32
}
```

**Parameters:**
- `layers`: List of integers specifying hidden layer sizes and output (last value must be 1)
- `lr`: Learning rate (default: `0.001` for uPBM/gcPBM, `1e-4` for SELEX)
- `normalization`: Enable input normalization (default: `true`)
- `num_epochs`: Number of training epochs (default: `100`)
- `batch_size`: Batch size for training (default: `32` for uPBM/gcPBM, `16` for SELEX)

If no config file is provided, default parameters are used. The config file can either have parameters at the top level or nested under `pytorch_model_params`.

### Output

Each script writes to `outputs/{method}/{protein}/{regressor}/` (gitignored), so different models for the same protein do not overwrite each other. Every run produces:
- the trained model (`model.sav` for uPBM/SELEX, `model.pck` for gcPBM)
- `predictions.png` — predicted vs ground-truth values for both the training and test split
- `feature_importances.png` and `top_features.png` — for models that provide importances (random forest)
- SELEX additionally keeps its intermediate training files next to the model

For PyTorch models, both the model state dict (`.pth`) and full model object (`.sav`/`.pck`) are saved.

## Quick Start Examples

```bash
# uPBM with default settings
python scripts/run_upbm.py --protein Gata4

# SELEX with custom regressor
python scripts/run_selex.py --protein Gata4 --cycle 4 --regressor random_forest

# gcPBM with custom config
python scripts/run_gcpbm.py --protein cbf1 --concentration 100 --config configs/dfn_config.json --regressor DFN
```
