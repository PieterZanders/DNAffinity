#!/bin/bash
# retrain DFN/CNN1D (set a) and CNN1D (set d) with weight saving for LRP; resumable
cd "$(dirname "$0")"
jobs() { for f in 0 1 2 3 4; do echo "cnn1d $f 1 lrp_a"; echo "dfn $f 1 lrp_a"; echo "cnn1d $f 0 lrp_d"; done; }
jobs | xargs -P 8 -L 1 bash -c 'C="{\"tf\":\"log\",\"mc\":10,\"mode\":\"perstrand\",\"ks\":\"4\",\"props\":\"avg+fce\",\"onehot\":1,\"shape\":$2,\"std\":1}"; OUTDIR=$3 SAVEW=1 SEEDS=0,1,2 NT=1 ../../.venv/bin/python train_final.py $0 "$C" $1 >> ../../outputs/selex_new_shape/queue_$3_$0.log 2>&1'
