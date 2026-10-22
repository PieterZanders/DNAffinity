#!/bin/bash
# shape-only sets (no one-hot): e=tetramer, f=hexamer, g=tetra+hexa; NN weights saved; resumable
cd "$(dirname "$0")"
jobs() { for s in "4 final_e" "6 final_f" "46 final_g"; do set -- $s
  for m in transformer cnnattention cnn1d dfn; do for f in 0 1 2 3 4; do echo "$m $f 0,1,2 $1 $2"; done; done
  for m in svr rf linear ridge; do echo "$m 0,1,2,3,4 0 $1 $2"; done; done; }
jobs | xargs -P 8 -L 1 bash -c 'C="{\"tf\":\"log\",\"mc\":10,\"mode\":\"perstrand\",\"ks\":\"$3\",\"props\":\"avg+fce\",\"onehot\":0,\"shape\":1,\"std\":1}"; OUTDIR=$4 SAVEW=1 SEEDS=$2 NT=1 ../../.venv/bin/python train_final.py $0 "$C" $1 >> ../../outputs/selex_new_shape/queue_$4_$0.log 2>&1'
