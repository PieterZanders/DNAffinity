#!/usr/bin/env python3
"""HT-SELEX regressor for affinity prediction.

Requires '{data_dir}/{protein}/{cycle}.txt' (preprocessed with the R pipeline
under notebooks/preprocess_selex/). Keeps k-mers with ObservedCount >= --min-count
(collapsing reverse complements), writes the training file to
outputs/selex/{protein}/{regressor}/, then trains.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from dnaffinity.common import (OUTPUTS_DIR, ROOT, SELEX_DFN_PARAMS, build_parser,
                               default_params, load_pytorch_params, save_model, train_model)
from dnaffinity.plotting import save_plots
from dnaffinity.preprocess import load_selex

parser = build_parser('SELEX regressor for affinity prediction',
                      default_data_dir=ROOT / 'data' / 'selex',
                      extra_args=[('--cycle', dict(type=str, required=True, help='Cycle number')),
                                  ('--min-count', dict(type=int, default=10,
                                                       help='Minimum ObservedCount per k-mer (default 10)'))])
args = parser.parse_args()

data_file = f'{args.data_dir}/{args.protein}/{args.cycle}.txt'
# fixed: keep k-mers with ObservedCount >= --min-count (reliable affinities), RC-collapsed.
# legacy: top 10% by background Probability (anti-correlated with affinity, see data audit)
df_train = load_selex(data_file, selection='legacy' if args.legacy_preprocessing else 'count',
                      min_count=args.min_count)
print(f'{len(df_train)} k-mers selected')

out_dir = OUTPUTS_DIR / 'selex' / args.protein / args.regressor
out_dir.mkdir(parents=True, exist_ok=True)
df_train.to_csv(out_dir / 'SELEX_training.txt', sep='\t', index=False)

chosen_features = ['presence_tetramer', 'avg', 'diagonal_fce']
# SELEX historically uses a deeper DFN; other models use their standard defaults
defaults = SELEX_DFN_PARAMS if args.regressor == 'DFN' else default_params(args.regressor)
pytorch_params = load_pytorch_params(args.config, defaults)

model = train_model(df_train, args, chosen_features, pytorch_params, label='the SELEX')

save_model(model, out_dir / 'model.sav', protocol=4)
importances = save_plots(model, out_dir)
if importances is not None:
    print(importances.head(10))
