# DNAffinity data audit (2026-10-20)

Script: `scripts/audit_data.py` writes `outputs/audit/audit.json` and the plots in `outputs/audit/*.png`.

## Tetramer tables (`data/input`)
| table | rows | ACGT tetramers | extra keys | NaN | RC consistency |
|---|---|---|---|---|---|
| avg_tetramer.dat (6 cols) | 290 | 256/256 (136 canonical) | 34 methylated (`M`/`g`) entries, unused | 0 | correct: Shift and Tilt change sign under RC, the other 4 columns are equal (e.g. AAAA/TTTT) |
| fce_tetramer.dat (36 cols) | 290 | 256/256 | same 34 | 0 | consistent with RC (off-diagonal Shift/Tilt couplings change sign) |
| electrostatic.txt (5 cols) | 256 | 256/256 | — | 0 | integer-valued; **parsed as strings by `Dataset`** (bug, fixed) |
| mgw_rohs.txt (1 col) | 256 | 256/256 | — | 0 | — |

* Coverage is complete and there are no duplicate or conflicting keys.
* **Feature scales differ a lot.** Twist is about 35, Rise about 3.3, Shift/Tilt are below 1, FCE diagonal terms run from about 0.02 (Tilt/Roll/Twist stiffness) to about 10 (Rise), presence counts are 0–7, and electrostatics run from -4 to 4. The code previously did no standardisation except LayerNorm inside the nets, so SVR in particular performed badly. Fix: StandardScaler fitted on the training set only.
* **Window alignment is correct.** Window *i* uses `seq[i:i+4]` for i = 0..L-4, so a 10-mer has 7 windows and a 30-mer has 27. The flat and windowed blocks match (covered by a test).

## uPBM Gata4 (`Gata4_training.txt`, 5,511 rows, all 10-mers, ACGT only)
* The file is the notebook output: MinMax-normalised, bin-undersampled, with top values oversampled.
* **557 exact duplicate sequences and 575 reverse-complement groups.** Labels within a group disagree: median spread is 0.072, while the median label is 0.11, and the duplicate pairs correlate at only r = 0.42. The same aligned 10-mer comes from different 35-mer probes. Under the old split these duplicates and RCs crossed train/test.
* Labels are right-skewed (skew 1.5).
* WEIGHT is 10 for only 45 rows (motif matches); their mean label is 0.31, vs 0.13 for the rest.
* `SELEX_test.txt` (in `data/selex/`) is **not SELEX data**. It shares 2,521 sequences and an identical label distribution with the uPBM training file (median 0.1107, skew 1.494). It looks like a mislabelled copy and should not be used as a SELEX test set.
* **Fix:** aggregate by canonical (RC-collapsed) sequence (mean label, max weight), giving 4,935 unique sequences. Splits are made on these groups.

## HT-SELEX Gata4 (`4.txt`, cycle 4, 126,624 10-mers)
* There are no exact duplicates. However, 17,376 k-mers have their RC present as a separate row with a different label.
* Most labels are noisy: 83% of k-mers have ObservedCount ≤ 5.
* **The original undersampling ("keep top 10% by `Probability`") is wrong.** `Probability` is the Markov-model *background* probability, and it is anti-correlated with Affinity (r = -0.38). The selected set has a median ObservedCount of 1, a median SE/Affinity of 1.41 (i.e. pure count-1 noise), and contains **none of the 100 highest-affinity k-mers**.
* **The enriched motif is ETS-like (`ACTTCCGG`), not GATA (`WGATAR`).** Mean affinity is 0.45 for k-mers containing TTCCGG and 0.24 for those containing GATA, against an overall mean of 0.23. This file may not be a Gata4 experiment, or it may be dominated by a contaminant or the library. It needs checking against the source.
* `selex.R` writes `6.txt` from round R6 with k = 10, but the repo ships cycle 4. The cycle choice is undocumented. The usual practice is the latest round with sufficient diversity, chosen by the max info-gain round from `selex.infogain`.
* **Fix:** select k-mers by `ObservedCount ≥ 10` (13,100 k-mers, so SE is controlled), collapse RCs by mean, and keep the legacy mode as an option for comparison.

## gcPBM cbf1 (`cbf1_100.txt`, 43,803 probes, 60 bp, trimmed to 30 bp)
* The IDs come in D/d/R/r variants: each 30-mer is measured **in both orientations and twice**. There are 43,803 rows but only **10,818 unique canonical sequences**, and 22,282 trimmed rows are exact duplicates.
  * Under the old pipeline, duplicates and RCs of test probes were in train. That is a second leakage source on top of the RC append.
  * The replicate agreement between two measurements of the same canonical sequence is r = 0.80 raw and r = 0.875 on log scale. **That is the practical ceiling for test Pearson.**
* Intensities are extremely heavy-tailed: they range from 507 to 222,933, with skew 7.4. Log-transforming the targets makes sense.
* **The flank filter is miscalibrated.** `cbf1_freq_matrix_6.txt` is a hard one-hot consensus `CACGTG`, so the "PWM score" is just the number of matching bases out of 6. The threshold `> 2.5` therefore discards any probe with ≥ 3/6 matches anywhere in the flanks.
  * It removes **77.5%** of probes. (My earlier figure of "55%" was a miscount: it compared counts after RC doubling.)
  * It would also remove **94.5% of random sequences**, so it is not detecting secondary sites.
  * Kept and discarded probes have essentially the same mean intensity (0.0114 vs 0.0125 normalised).
  * Only the forward strand is scanned. That is fine here because CACGTG is palindromic.
  * **Fix:** discard only flanks with ≥ 5/6 matches to the E-box (threshold 4.5). This removes 3.6% of canonical sequences (394 of 10,818).
* MinMax normalisation was computed on all probes, including test. Fix: compute target scaling in the model on train only.

## Summary of the fixes applied in code (`dnaffinity/preprocess.py`, `dataset.py`, `model.py`)
1. Collapse each dataset to unique canonical sequences, averaging labels. The RC is then added back as augmentation, with **both strands in the same split group**.
2. Seeded GroupShuffleSplit into train/val/test, plus 5-fold shuffled GroupKFold for CV.
3. Cast electrostatics to float. Scale features and targets on train only, optionally with a log target.
4. SELEX: select by count (≥10) instead of top background probability.
5. gcPBM: flank threshold 4.5 (≥5/6 E-box matches) instead of 2.5.
6. Data provenance issues that need a human (not fixable in code): the SELEX "Gata4" file shows an ETS motif, and `SELEX_test.txt` is a uPBM copy.
