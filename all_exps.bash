#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./all_exps.bash <script> <seeds>

set -euo pipefail

# params
script="${1:-}"
environment="${2:-"kitchen"}"
seeds="${3:-1}"

if [[ -z "$script" ]]; then
  echo "No script provided. Exiting."
  exit 1
fi

# main script logic goes here
echo "Starting script with argument: $script, $seeds"

# create a list of tasks
kitchen_tasks=("kitchen-mixed-v0" "kitchen-partial-v0" "kitchen-complete-v0")
antmaze_tasks=("antmaze-large-diverse-v2" "antmaze-large-play-v2" "antmaze-medium-diverse-v2" "antmaze-medium-play-v2")
mujoco_tasks=("halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-v2"
              "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2"
              "walker2d-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-medium-v2")

# check if task is provided and set tasks accordingly

if [[ "$environment" == "kitchen" ]]; then
  tasks=( "${kitchen_tasks[@]}" )
elif [[ "$environment" == "antmaze" ]]; then
  tasks=( "${antmaze_tasks[@]}" )
elif [[ "$environment" == "mujoco" ]]; then
  tasks=( "${mujoco_tasks[@]}" )
else
  echo "No valid task provided. Exiting."
  exit 1
fi

#==============================#
#          Plain Runs          #
#==============================#

for ((i=0; i<seeds; i++)); do
  for task in "${tasks[@]}"; do
    echo "Running plain run for seed $i on $environment:$task"
    python "$script" env_name="$environment" task="$task" loss_type="all" exp_name="plain_run_${i}_${task}" seed="$i" eval_interval=500 > "logs/plain_run_${i}_${task}.log" 2>&1 &
  done
done

#===============================#
#         Jacobian Runs         #
#===============================#
# jacobian_weights=(50 200)

# for ((i=0; i<seeds; i++)); do
#   for task in "${tasks[@]}"; do
#     for weight in "${jacobian_weights[@]}"; do
#       python "$script" env_name="$environment" task="$task" loss_type="all" loss_weights.jacobian="$weight" exp_name="jacobian_run_${i}_${weight}_${task}" seed="$i" &
#     done
#   done
#   wait
# done

#===============================#
#         Lambda Runs           #
#===============================#

# lambda_contractions=(0.01 0.05 0.1 0.2 0.4)

# for ((i=0; i<seeds; i++)); do
#   for task in "${tasks[@]}"; do
#       for lambda in "${lambda_contractions[@]}"; do
#         python "$script" env_name="$environment" task="$task" loss_type="all" lambda_contraction="$lambda" exp_name="lambda_run_${i}_${lambda}" seed="$i" &
#   done
# done

# #===============================#
# #          Eigen Runs           #
# #===============================#
# # your code here

echo "Script finished."
