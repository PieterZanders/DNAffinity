#!/bin/bash
# permuted-MD-table control on (e) tetramer shape-only: RF, DFN, CNN1D; 2 permutations
cd "$(dirname "$0")"
jobs() { for p in 1 2; do for m in cnn1d dfn; do for f in 0 1 2 3 4; do echo "$m $f 0,1,2 $p"; done; done; echo "rf 0,1,2,3,4 0 $p"; done; }
jobs | xargs -P 8 -L 1 bash -c 'C="{\"tf\":\"log\",\"mc\":10,\"mode\":\"perstrand\",\"ks\":\"4\",\"props\":\"avg+fce\",\"onehot\":0,\"shape\":1,\"std\":1}"; PERM_TAB=$3 OUTDIR=final_e_perm$3 SEEDS=$2 NT=1 ../../.venv/bin/python train_final.py $0 "$C" $1 >> ../../outputs/selex_new_shape/queue_perm$3_$0.log 2>&1'
