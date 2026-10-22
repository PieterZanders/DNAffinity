# SELEX Gata4 + new SELEX shape tables (branch selex-new-shape)

## Data characterization (outputs/selex_new_shape/characterization.txt)
- The new `data/selex/{tetramer,hexamer}/{avg,fce}_*.dat` files contain **features only**, with no labels. avg = 6 helical params (SHIFT..TWIST); fce = flattened symmetric 6x6 stiffness matrix (only 21 unique values; all eigenvalues > 0). They cover every k-mer (256 / 4096), with no NaNs. The avg files are tab-separated with a `SEQUENCE` header; the fce files are space-separated. Labels still come from `Gata4/4.txt`.
- The new tetramer tables **differ from data/input**. Per-column correlation for avg is 0.83–0.98 (SHIFT 0.83, TILT 0.85); for the fce diagonal it is 0.90–0.97. They are a different parameter set, not a copy.
- RC consistency holds: x vs RC(x), with SHIFT/TILT sign-flipped, gives r ≥ 0.998.
- Hexamer avg values correlate 0.96–0.99 with the avg of their central tetramer, so they add little new information.
- `dnaffinity.Dataset` hard-codes `data/input`, so these features were built by a separate loader (`scripts/selex_new_shape/lib.py`). `notebooks/train_and_test.ipynb` only shows the uPBM workflow (Dataset/Model, onehot/presence + avg + diag fce) and does not use the SELEX files.
- 4.txt (re-verified):
  - It has 126,624 of the 1,048,576 10-mers (12.1%); 52.5% have count 1; 13,100 have count ≥10.
  - Affinity = 0.1720·(obs/exp)^0.25 exactly (sd 2e-16). Median SE·√n/Aff = 1.414.
  - Forward vs RC log affinity: r = 0.417 / 0.412 / 0.425 at count ≥1 / 10 / 30.
- **SELEX_test.txt does not match 4.txt.** Only 8.4% of its 10-mers are in 4.txt. On the 462 that overlap, its VALUE is uncorrelated with 4.txt affinity (Pearson of logs −0.03; Spearman −0.06). Its max is 1.0 and WEIGHT is 1 or 10. It was **not used** as a test set. The test set is a held-out 10% of RC-groups from 4.txt.

## Protocol
- Train pool: 5-fold GroupKFold with groups = k-mer + its reverse complement. Test: a fixed 10% of RC groups, the same for every config.
- Ablation metric: R² on log affinity, scored on held-out-fold rows that have a measured per-strand label with count ≥10. Every config is scored on this same set.
- Feature windows: 7 tetramer windows per 10-mer. Hexamers are aligned on their central tetramer and zero-padded for the CNN.

## Ablation (CV val R², mean±sd over 5 folds) – see ablation_results.csv
| change vs base (log, count≥10, per-strand, tetramer avg+fce, std) | RF | CNN1D |
|---|---|---|
| base | 0.588±0.015 | 0.692±0.018 |
| raw label | 0.579 | 0.691 |
| count≥1 / ≥3 / ≥30 | −1.449 / 0.044 / −0.227 | – / – / 0.158 |
| merged labels (+RC aug) | 0.438 | 0.520 |
| hexamer only / tetra+hexa | 0.546 / 0.579 | 0.578 / 0.672 |
| avg only / fce only | 0.560 / 0.586 | 0.661 / 0.680 |
| no standardization | 0.588 | 0.653 |
| + one-hot | **0.599** | **0.738** |
| one-hot only (no shape) | 0.563 | 0.739 |
| + one-hot, tetra+hexa | 0.589 | 0.721 |

Chosen: log affinity, count ≥10, per-strand labels, tetramer avg+fce(21) + window one-hot, z-score standardization fit on the training fold.

## Final results (train/val = CV folds; test = held-out; mean±sd over folds×seeds)
| model | runs | train R² | val R² | test R² | test Pearson | test Spearman | gap (train−test) | test ensemble R² |
|---|---|---|---|---|---|---|---|---|
| RF | 5 | 0.884±0.001 | 0.599±0.015 | 0.572±0.004 | 0.762±0.003 | 0.744±0.003 | 0.311 | 0.581 |
| DFN | 15 | 0.820±0.026 | 0.660±0.011 | 0.644±0.011 | 0.803±0.007 | 0.791±0.008 | 0.176 | 0.673 |
| CNN1D | 15 | 0.889±0.032 | 0.735±0.015 | 0.736±0.012 | 0.860±0.006 | 0.844±0.007 | 0.153 | 0.788 |
Full Pearson/Spearman for every split: results_final.csv.

Overfitting:
- RF memorizes noisy labels (gap 0.31), but val and test agree.
- DFN: best epoch about 44 of 70; train MSE 0.19 vs val 0.33. Mild overfitting, controlled by early stopping.
- CNN1D: the val curve plateaus at about epoch 10–30 (best ≈27) while train MSE keeps falling to 0.07. It overfits after the plateau; early stopping restores the best epoch. Val and test agree (0.735 vs 0.736), so there is no leakage.

Plots: plots/{rf,dfn,cnn1d}_truth_vs_pred.png, plots/{dfn,cnn1d}_learning_curves.png
