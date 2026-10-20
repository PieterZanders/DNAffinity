# DNAffinity: fixed evaluation and deep-model improvements (INTERIM, 2026-10-20 16:55 CEST)

**Status:** gcPBM is complete (all 41 runs). SELEX is mostly complete: all fixed-preprocessing deep and linear runs are done, while RF variants, Transformer-orig and the legacy-preprocessing runs are still running. uPBM is queued in the same resumable suite and has not started yet. Rerun `scripts/summarize_cv.py` to refresh `fixed_results.csv` (all metric columns, including RMSE and train R2).

## Protocol
* **Splits:** 5-fold shuffled GroupKFold (seed 0). A group is one canonical sequence, so a sequence, its reverse complement and any duplicates always share a fold.
* **Validation:** neural models hold out 10% of the training groups for early stopping. Max 200 epochs, patience 20, AdamW (wd 1e-4), ReduceLROnPlateau, batch 64, shuffled batches, seeded.
* **Scaling:** features and targets are standardised on the training rows only. Metrics are computed on the original label scale.
* **Variants (cumulative):** `base` → `rcavg` (predictions averaged over both strands; metrics per unique sequence) → `onehot` (one-hot sequence channels: 16 per tetramer window for CNN/attention/Transformer, flat one-hot for the others) → `loghuber` (log targets, plus Huber loss for neural nets). `orig` is the Transformer with its original hyperparameters (lr 1e-3, CLS pooling, d_model 64, no warmup). `legacy` uses the original preprocessing but the fixed evaluation.
* **Hardware:** CPU only, 6–8 jobs in parallel with 1 thread each. Times are wall-clock per 5-fold CV.

## Data audit, key points (details in `data_audit.md`)
1. **Leakage in the old evaluation:** reverse complements were appended after the originals and the split did not shuffle, so 100% of test rows had their other strand in train. The data also contains duplicates: uPBM has 557 exact duplicates and 575 RC groups (duplicate labels agree only at r = 0.42), and gcPBM measures every 30-mer about 4 times (2 replicates × 2 orientations), giving only 10,818 unique sequences out of 43,803 probes.
2. **gcPBM flank filter was miscalibrated.** The "PWM" is a hard CACGTG consensus, and the threshold of 2.5 means "≥3/6 matching bases anywhere in the flank". It removed 77.5% of probes, and would remove 94.5% of random sequences. Fixed to ≥5/6 matches, which removes 3.6%. Intensities are very heavy-tailed (skew 7.4), and the replicate ceiling is r = 0.80 raw / 0.875 log.
3. **SELEX selection was inverted.** The old code kept the top 10% by background *Probability*, which is anti-correlated with affinity (r = −0.38). The kept k-mers had a median count of 1 and included none of the 100 highest-affinity k-mers. Fixed by keeping k-mers with ObservedCount ≥ 10. **Also, the enriched motif in the "Gata4" SELEX file is ETS-like (ACTTCCGG), not GATA.** That is a provenance question for a human. Separately, `SELEX_test.txt` is a copy of the uPBM data.
4. Electrostatics were parsed as strings. Feature scales differ by orders of magnitude, and nothing was standardised. RF had only 10 trees and ridge used alpha = 1e-4 (effectively OLS). The tetramer tables themselves are complete and RC-consistent, and the feature windows align correctly.

## gcPBM cbf1: before vs after
Before = the leaky single split from `baseline_results.csv` (30 epochs). After = 5-fold group CV (mean ± std). On gcPBM, Pearson and R2 are computed on raw intensities and are dominated by the few very bright probes, so **Spearman is the more informative metric**.

| model | before: r / ρ / R2 (leaky split) | legacy prep + fixed eval | fixed prep, base | best fixed variant (Pearson / Spearman / R2) |
|---|---|---|---|---|
| CNNAttention | 0.907 / 0.505 / 0.807 | 0.914±0.025 / 0.471±0.048 / 0.828±0.049 | 0.974±0.006 / 0.666±0.026 / 0.944±0.011 | **loghuber: 0.979±0.004 / 0.847±0.012 / 0.956±0.009** |
| CNN1D | 0.906 / 0.640 / 0.821 | 0.914±0.014 / 0.518±0.032 / 0.827±0.030 | 0.975±0.006 / 0.687±0.013 / 0.948±0.010 | loghuber: 0.970±0.006 / 0.838±0.011 / 0.936±0.018 |
| Transformer | constant output (R2 ≈ 0) | 0.853±0.032 / 0.396±0.023 / 0.713±0.063 | 0.934±0.010 / 0.505±0.060 / 0.867±0.023 | loghuber: 0.961±0.006 / 0.812±0.004 / 0.922±0.010 |
| DFN | 0.848 / 0.458 / 0.719 | 0.829±0.060 / 0.460±0.057 / 0.684±0.104 | 0.917±0.021 / 0.620±0.047 / 0.841±0.039 | loghuber: 0.912±0.012 / 0.773±0.009 / 0.829±0.023 |
| random_forest | 0.882 / 0.517 / 0.777 | 0.876±0.057 / 0.548±0.027 / 0.768±0.098 | 0.923±0.017 / 0.701±0.014 / 0.850±0.030 | loghuber: 0.929±0.020 / 0.725±0.017 / 0.852±0.034 |
| svr | 0.554 / 0.414 / −0.651 | 0.712±0.069 / 0.456±0.028 / 0.436±0.075 | 0.771±0.022 / 0.557±0.030 / 0.452±0.017 | loghuber: 0.845±0.023 / 0.685±0.021 / 0.401±0.040 |
| ridge / linear | crashed (string features) | 0.631±0.027 / 0.372±0.031 / 0.365±0.066 | 0.651±0.023 / 0.449±0.032 / 0.420±0.033 | loghuber: 0.800±0.013 / 0.537±0.034 / 0.402±0.022 |

**Best gcPBM model: CNNAttention with one-hot channels, RC averaging, log targets and Huber loss.** It reaches Spearman 0.847 ± 0.012 and Pearson 0.979 ± 0.004. CNN1D is close behind and about 1.5× cheaper (Spearman 0.838).

**What helped on gcPBM, ranked by Spearman gain**
* **Fixed preprocessing:** +0.15 to +0.20 for every model (legacy → fixed base). Training on 10.4k unique sequences instead of 2.4k after the over-aggressive filter, with replicate-averaged labels, matters most.
* **Log target + Huber:** +0.13 to +0.28 for the neural nets (e.g. CNN1D 0.715 → 0.838, Transformer 0.560 → 0.812). Pearson/R2 on the raw scale change little because they are dominated by outliers.
* **RC averaging:** small but consistent, +0.03 to +0.04 for the CNNs and DFN. Essentially zero for the sklearn models.
* **Transformer fix:** the original configuration no longer collapses once the pipeline is fixed (Spearman 0.494 orig vs 0.505 base; R2 is actually higher for orig, 0.898 vs 0.867). The large gain comes from log/Huber. The Transformer is still about 2–3× slower than the CNNs and does not beat them.

**What did not help on gcPBM**
* One-hot channels were neutral: CNN1D 0.719 → 0.715 (±0.04) and CNNAttention 0.705 → 0.686. The 30-mer shape and physics features already encode the E-box core.
* Log targets hurt R2 for linear models and SVR, because the back-transformation inflates extremes.

## SELEX (partial; fixed preprocessing, count ≥ 10, 10,771 unique 10-mers)
| model | base | rcavg | onehot | loghuber |
|---|---|---|---|---|
| CNN1D | 0.735 / 0.648 / 0.528 | 0.753 / 0.669 / 0.560 | **0.803±0.011 / 0.736±0.015 / 0.644±0.017** | 0.796 / 0.732 / 0.626 |
| CNNAttention | 0.723 / 0.628 / 0.516 | 0.744 / 0.653 / 0.550 | 0.798 / 0.730 / 0.634 | 0.786 / 0.725 / 0.611 |
| Transformer | 0.699 / 0.597 / 0.485 | 0.707 / 0.608 / 0.498 | 0.754 / 0.682 / 0.568 | 0.747 / 0.685 / 0.549 |
| DFN | 0.710 / 0.627 / 0.494 | 0.727 / 0.648 / 0.524 | 0.733 / 0.654 / 0.534 | 0.722 / 0.662 / 0.507 |
| svr | 0.723 / 0.667 / 0.516 | 0.723 / 0.667 / 0.516 | 0.724 / 0.666 / 0.517 | 0.731 / 0.666 / 0.525 |
| random_forest | 0.682 / 0.572 / 0.464 | (running) | (running) | (running) |
| ridge / linear | 0.558 / 0.535 / 0.310 | same | same | 0.560 / 0.544 / 0.296 |

Values are Pearson / Spearman / R2, fold std ≈ 0.01. **Current SELEX leader: CNN1D + RC averaging + one-hot** (Pearson 0.803, R2 0.644). On SELEX, one-hot channels are the main gain (+0.05 Pearson, +0.07 Spearman for the windowed models), RC averaging adds about +0.02, and log/Huber is neutral to slightly negative. The SELEX numbers are not comparable to the old baseline (R2 0.53, CNNAttention), because the old data were a different, noise-dominated k-mer set evaluated with leakage. The legacy-preprocessing runs for this comparison are still pending.

## uPBM
Queued and not yet started. A rough estimate is about 1.5–2 h for the remaining SELEX and uPBM runs together at 6 parallel jobs (uPBM is the smallest dataset; RF and SVR are the slowest jobs).
