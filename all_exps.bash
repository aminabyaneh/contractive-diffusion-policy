#!/bin/bash

# cd_edp_kitchen.bash
# description: Run batched experiments with

set -euo pipefail

script="${1:-}"
seeds="${2:-1}"
task="${3:-}"

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
if [[ "$task" == "kitchen" ]]; then
  tasks=("${kitchen_tasks[@]}")
elif [[ "$task" == "antmaze" ]]; then
  tasks=("${antmaze_tasks[@]}")
elif [[ "$task" == "mujoco" ]]; then
  tasks=("${mujoco_tasks[@]}")
else
  echo "No valid task provided. Exiting."
  exit 1
fi


#==============================#
#          Plain Runs          #
#==============================#

for ((i=0; i<seeds; i++)); do
  for task in "${tasks[@]}"; do
    python "$script" loss_type="all" exp_name="plain_run_${i}" seed="$i" task="$task" &
done

#===============================#
#         Jacobian Runs         #
#===============================#
# jacobian_weights=(200 100 50)

# for ((i=0; i<seeds; i++)); do
#   for task in "${tasks[@]}"; do
#     for weight in "${jacobian_weights[@]}"; do
#       python "$script" loss_type="all" loss_weights.jacobian="$weight" exp_name="jacobian_run_${i}_${weight}" seed="$i" task="$task" &
#     done
#   done
# done

#===============================#
#         Lambda Runs           #
#===============================#

#===============================#
#          Eigen Runs           #
#===============================#
# your code here
echo "Script finished."