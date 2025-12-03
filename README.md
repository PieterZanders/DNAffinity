# DNAffinity

#### Machine Learning method to predict binding sites for different Transcription Factors using data from different experimental techniques.

DNAffinity presents a physics-based machine learning approach to predict *in vitro* 
transcription factor binding affinities from structural and mechanical DNA properties 
directly derived from atomistic molecular dynamics simulations. The method is able to 
predict affinities obtained with high-throughput techniques as different as uPBM, gcPBM 
and HT-SELEX. When complemented with chromatin structure information, our *in vitro* 
trained method provides also good estimates of *in vivo* binding sites in yeast.

## Installation

---------------

Required dependencies:
- Python 3.x
- NumPy, Pandas, Matplotlib
- scikit-learn
- PyTorch (for `DFN`)

Install PyTorch from [pytorch.org](https://pytorch.org/) based on your system.

## Usage

---------------

The pipeline for a specific protein consists of:

1. Preprocessing the data (specific steps for each experimental technique; details on upbm and selex can be found under the **notebooks** folder)
2. Getting the preprocessed data file (e.g. the `Gata4_training.txt` file from the **test_data** folder) 
3. Running the corresponding regressor with command-line arguments

### Required Files

Make sure you have all the required files (compare formatting with the templates): 
- `{data_dir}/{protein}_training.txt` for `upbm_regressor.py`
- `{data_dir}/{cycle}.txt` for `selex_regressor.py`
- `{data_dir}/{protein}_{concentration}.txt` and `{data_dir}/{protein}_freq_matrix_6.txt` for `gcpbm_regressor.py`

### Command-Line Interface

All regressor scripts support command-line arguments for flexible configuration:

#### uPBM Regressor

```bash
python upbm_regressor.py PROTEIN [OPTIONS]
```

**Arguments:**
- `--PROTEIN` (required): Protein name

**Options:**
- `--data-dir PATH`: Data directory (default: `test_data/{protein}`)
- `--regressor TYPE`: Regressor type - `random_forest`, `linear`, `ridge`, `svr`, `DFN` (default: `DFN`)
- `--training-set-size FLOAT`: Training set size fraction (default: `0.9`)
- `--randomize-fce`: Randomize FCE features
- `--score STRING`: Score column name (default: `Median_intensity`)
- `--model-target TYPE`: Model target - `octamers` or `tetramers` (default: `octamers`)
- `--config PATH`: Path to JSON file with PyTorch model parameters

**Example:**
```bash
python upbm_regressor.py Gata4 --regressor DFN --config config.json
```

#### SELEX Regressor

```bash
python selex_regressor.py --protein PROTEIN --cycle CYCLE [OPTIONS]
```

**Arguments:**
- `--PROTEIN` (required): Protein name
- `--CYCLE` (required): Cycle number

**Options:** Same as uPBM regressor

**Example:**
```bash
python selex_regressor.py 
--protein Gata4 
--cycle 4 
--regressor DFN 
--training-set-size 0.9
```

#### gcPBM Regressor

```bash
python gcpbm_regressor.py --protein PROTEIN --concentration CONCENTRATION [OPTIONS]
```

**Arguments:**
- `--PROTEIN` (required): Protein name
- `--CONCENTRATION` (required): Concentration value

**Options:** Same as uPBM regressor

**Example:**
```bash
python gcpbm_regressor.py cbf1 100 --regressor DFN --config config.json
```

### Regressor Types

Available regressor types:

- **`random_forest`**: Random Forest regressor (scikit-learn)
- **`linear`**: Linear Regression (scikit-learn)
- **`ridge`**: Ridge Regression (scikit-learn)
- **`svr`**: Support Vector Regression (scikit-learn)
- **`DFN`**: PyTorch feedforward neural network (recommended)

### PyTorch Model Configuration

For `DFN` regressors, you can specify model parameters via a JSON config file:

**Example `config.json`:**
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

Trained models are saved to:
- `trained_models/{protein}_finalized_model.sav` (for uPBM and SELEX)
- `output_gcpbm/{protein}/model.pck` (for gcPBM)

For PyTorch models, both the model state dict (`.pth`) and full model object (`.sav`/`.pck`) are saved.

## Quick Start Examples

```bash
# uPBM with default settings
python upbm_regressor.py Gata4

# SELEX with custom regressor
python selex_regressor.py Gata4 4 --regressor random_forest

# gcPBM with custom config
python gcpbm_regressor.py cbf1 100 --config my_config.json --regressor DFN
```
