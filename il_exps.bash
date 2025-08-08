#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./all_exps.bash <script> <seeds>

set -euo pipefail

# params
environment="${1:-"robomimic_image"}"
seeds="${2:-5}"

# create logs directory if it doesn't exist
mkdir -p logs

# main script logic goes here
echo "Starting script with argument: $environment, $seeds"

tasks=("lift" "can") # ("square" "transport")
jacobian_penalty=(0.1 1 10) # 0.0
available_data_rate=(1.0 0.5 0.1) # 0.1 0.05
gradient_steps=200000
eval_freq=20000
project="cdp_insufficient_data_image"

for jp in "${jacobian_penalty[@]}"; do
  for task in "${tasks[@]}"; do
    for adr in "${available_data_rate[@]}"; do
      for i in $(shuf -i 0-9999 -n $seeds); do
        echo "Running $task seed $i with penalty $jp and available data rate $adr"

        if [[ "$environment" == "robomimic_lowdim" ]]; then
          python dbc_robomimic.py task="$task" loss_type="jacobian" exp_name="jrun_${i}_${task}_${jp}_${adr}" seed="$i" project="$project" gradient_steps="$gradient_steps" eval_freq="$eval_freq" loss_weights.jacobian="$jp" available_data_rate="$adr" > "logs/jrun_${i}_${task}_${jp}_${adr}.log" 2>&1 &
        elif [[ "$environment" == "robomimic_image" ]]; then
          python dbc_robomimic_image.py task="$task" loss_type="jacobian" exp_name="jrun_${i}_${task}_${jp}_${adr}" seed="$i" project="$project" gradient_steps="$gradient_steps" eval_freq="$eval_freq" loss_weights.jacobian="$jp" available_data_rate="$adr" > "logs/jrun_${i}_${task}_${jp}_${adr}.log" 2>&1 &
        fi

      done
      wait
    done
  done
done

echo "Script finished."
