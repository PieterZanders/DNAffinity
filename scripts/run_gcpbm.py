#!/usr/bin/env python3
"""gcPBM regressor for affinity prediction.

Requires '{data_dir}/{protein}/{protein}_{concentration}.txt' (raw intensities)
and '{data_dir}/{protein}/{protein}_freq_matrix_6.txt' (position frequency
matrix). Probes whose flanking regions score too high against the frequency
matrix (secondary binding sites) are discarded before training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from dnaffinity.common import (OUTPUTS_DIR, ROOT, build_parser, default_params,
                               load_pytorch_params, save_model, train_model)
from dnaffinity.plotting import save_plots
from dnaffinity.preprocess import load_gcpbm

parser = build_parser('gcPBM regressor for affinity prediction',
                      default_data_dir=ROOT / 'data' / 'gcpbm',
                      extra_args=[('--concentration', dict(type=str, required=True,
                                                           help='Concentration label')),
                                  ('--flank-threshold', dict(type=float, default=4.5,
                                                             help='Max flank score vs the frequency matrix '
                                                                  '(default 4.5 = discard >=5/6 E-box matches)'))])
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Preprocessing: normalize intensities and filter out probes with secondary
# binding sites in the flanks
# ---------------------------------------------------------------------------
data_file = f'{args.data_dir}/{args.protein}/{args.protein}_{args.concentration}.txt'
fm_file = f'{args.data_dir}/{args.protein}/{args.protein}_freq_matrix_6.txt'
if args.legacy_preprocessing:
    # original behaviour: no replicate aggregation, flank threshold 2.5 (drops ~78%)
    df_train = load_gcpbm(data_file, fm_file, flank_threshold=2.5, aggregate=False)
else:
    df_train = load_gcpbm(data_file, fm_file, flank_threshold=args.flank_threshold)
print(f'{len(df_train)} probes kept after flank filtering / aggregation')

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
chosen_features = ['presence_tetramer', 'avg', 'diagonal_fce', 'electrostatic']
pytorch_params = load_pytorch_params(args.config, default_params(args.regressor))

model = train_model(df_train, args, chosen_features, pytorch_params, label='the gcPBM')

out_dir = OUTPUTS_DIR / 'gcpbm' / args.protein / args.regressor
save_model(model, out_dir / 'model.pck')
importances = save_plots(model, out_dir)
if importances is not None:
    print(importances.head(10))
else:
    print("Feature importance analysis not available for this regressor type")
