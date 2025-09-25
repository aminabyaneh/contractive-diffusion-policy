#!/usr/bin/env bash

set -euo pipefail

# parameters
environment="${1:-"robomimic_lowdim"}"
seeds="${2:-1}"

# create logs directory if it doesn't exist
mkdir -p logs

echo "Starting script with environment: $environment, seeds: $seeds"

# configuration
tasks=("lift" "can" "square" "transport")
jacobian_penalty=(0.001 0.01 0.1 0.0 1.0 10.0 100.0)
available_data_rate=(1.0 0.5 0.1 0.05)
loss_type="jacobian"
gradient_steps=100000
eval_freq=20000
project="il_cdp"
diffusion_network="chi_transformer"

# main execution loop
for jp in "${jacobian_penalty[@]}"; do
  for task in "${tasks[@]}"; do
    for adr in "${available_data_rate[@]}"; do
      for i in $(shuf -i 0-9999 -n $seeds); do
        echo "Running task: $task, seed: $i, penalty: $jp, data rate: $adr"

        # determine script based on environment
        if [[ "$environment" == "robomimic_lowdim" ]]; then
          python baselines/dp.py \
            task="$task" \
            loss_type="$loss_type" \
            exp_name="jrun_${i}_${task}_${jp}_${adr}" \
            seed="$i" \
            project="$project" \
            gradient_steps="$gradient_steps" \
            eval_freq="$eval_freq" \
            nn="$diffusion_network" \
            loss_weights.jacobian="$jp" \
            available_data_rate="$adr" \
            > "logs/jrun_${i}_${task}_${jp}_${adr}.log" 2>&1 &

        elif [[ "$environment" == "robomimic_image" ]]; then
          python baselines/dp.py \
            task="$task" \
            loss_type="$loss_type" \
            exp_name="jrun_${i}_${task}_${jp}_${adr}" \
            seed="$i" \
            project="$project" \
            gradient_steps="$gradient_steps" \
            eval_freq="$eval_freq" \
            nn="$diffusion_network" \
            loss_weights.jacobian="$jp" \
            available_data_rate="$adr" \
            > "logs/jrun_${i}_${task}_${jp}_${adr}.log" 2>&1 &
        fi
      done
      wait
    done
  done
done

echo "Script finished."
