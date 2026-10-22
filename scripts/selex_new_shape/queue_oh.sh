#!/bin/bash
# one-hot-only feature set (no shape), final protocol; resumable
cd "$(dirname "$0")"
jobs() { for m in linear ridge svr rf; do echo "$m 0,1,2,3,4 0"; done
         for m in transformer cnnattention cnn1d dfn; do for f in 0 1 2 3 4; do echo "$m $f 0,1,2"; done; done; }
jobs | xargs -P 8 -L 1 bash -c 'C="{\"tf\":\"log\",\"mc\":10,\"mode\":\"perstrand\",\"ks\":\"4\",\"props\":\"avg+fce\",\"onehot\":1,\"shape\":0,\"std\":1}"; OUTDIR=final_onehot SEEDS=$2 NT=1 ../../.venv/bin/python train_final.py $0 "$C" $1 >> ../../outputs/selex_new_shape/queue_onehot_$0.log 2>&1'
