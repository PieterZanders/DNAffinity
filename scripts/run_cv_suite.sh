#!/usr/bin/env bash
# Resumable 5-fold CV suite. Skips jobs whose result JSON already exists.
# Usage: nohup scripts/run_cv_suite.sh [PARALLEL] > outputs/cv_runs/suite.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
P=${1:-8}
OUT=outputs/cv_runs
mkdir -p "$OUT/logs"
jobs() {
  for d in gcpbm selex upbm; do
    for m in CNNAttention CNN1D Transformer DFN svr random_forest ridge linear; do
      for v in base rcavg onehot loghuber; do echo "$d fixed $m $v"; done
    done
    echo "$d fixed Transformer orig"
    for m in CNNAttention CNN1D Transformer DFN svr random_forest ridge linear; do echo "$d legacy $m base"; done
  done
}
jobs | while read -r d p m v; do
  [ -f "$OUT/${d}_${p}_${m}_${v}.json" ] || echo "$d $p $m $v"
done | xargs -P "$P" -L 1 sh -c \
  '.venv/bin/python scripts/benchmark.py $0 $1 $2 $3 --out '"$OUT"' > '"$OUT"'/logs/$0_$1_$2_$3.log 2>&1; echo "done $0 $1 $2 $3 rc=$?"'
echo "suite finished"
