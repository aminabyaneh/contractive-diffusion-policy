#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./all_exps.bash <script> <seeds>

set -euo pipefail
seeds=4

# gradient_steps=350000
# env=pusht

jacobian=(100)

gradient_steps=600000
env=pusht_image

# condition based on environment
if [ "$env" == "pusht" ]; then
  for j in {0..3}; do
    for ((i=0; i<seeds; i++)); do
      echo "Running plain run for seed $i on pusht with jacobian ${jacobian[j]}"
      python contractive_dbc_pusht.py  gradient_steps=$gradient_steps seed=$i loss_weights.jacobian=${jacobian[j]} lambda_contr=0.001 > logs/output/plain_run_${i}_${jacobian[j]}_pusht.log 2>&1 &
    done
    wait
  done
elif [ "$env" == "pusht_image" ]; then
  echo "Running plain run for pusht_image"
  for j in {0..3}; do
    for ((i=0; i<seeds; i++)); do
      echo "Running plain run for seed $i on pusht_image with jacobian ${jacobian[j]}"
      python contractive_dbc_pusht_image.py gradient_steps=$gradient_steps seed=$i loss_weights.jacobian=${jacobian[j]} > logs/output/plainimage_run_${i}_${jacobian[j]}_pusht_image.log 2>&1 &
    done
    wait
  done
fi

echo "Script finished."
