#!/bin/bash
# job list: "model ks folds outdir seeds"; resumable (skips finished fold/seed files). 8 parallel single-thread workers.
cd "$(dirname "$0")"
jobs() { for ks in 6 46; do for m in linear ridge svr rf lightgbm; do echo "$m $ks 0,1,2,3,4 final_ks$ks 0"; done; done
         for m in linear ridge svr rf lightgbm; do echo "$m 4 0,1,2,3,4 final_retime_ks4 0"; done
         for ks in 6 46; do for m in transformer cnnattention cnn1d dfn; do for f in 0 1 2 3 4; do echo "$m $ks $f final_ks$ks 0,1,2"; done; done; done
         for m in transformer cnnattention cnn1d dfn; do for f in 0 1 2 3 4; do echo "$m 4 $f final_retime_ks4 0"; done; done; }
jobs | xargs -P 8 -L 1 bash -c 'C="{\"tf\":\"log\",\"mc\":10,\"mode\":\"perstrand\",\"ks\":\"$1\",\"props\":\"avg+fce\",\"onehot\":1,\"shape\":1,\"std\":1}"; OUTDIR=$3 SEEDS=$4 NT=1 ../../.venv/bin/python train_final.py $0 "$C" $2 >> ../../outputs/selex_new_shape/queue_$3_$0.log 2>&1'
