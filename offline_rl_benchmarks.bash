#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./offline_rl_benchmarks.bash [environment] [seeds]
# example: ./offline_rl_benchmarks.bash kitchen 5

set -euo pipefail

# params
environment="${1:-"kitchen"}"
seeds="${2:-1}"

# create logs directory if it doesn't exist
mkdir -p logs

# main script logic goes here
echo "Starting script with argument: $environment, $seeds"

# create a list of tasks, grouped by environment
antmaze_m_tasks=("antmaze-medium-play-v2" "antmaze-medium-diverse-v2")
antmaze_l_tasks=("antmaze-large-play-v2" "antmaze-large-diverse-v2")
kitchen_tasks=("kitchen-mixed-v0" "kitchen-partial-v0" "kitchen-complete-v0")
mujoco_m_tasks=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2")
mujoco_me_tasks=("hopper-medium-expert-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-expert-v2")
mujoco_mr_tasks=("halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2")

# check if task is provided and set tasks accordingly
if [[ "$environment" == "kitchen" ]]; then
  tasks=( "${kitchen_tasks[@]}" )
  env="kitchen"
elif [[ "$environment" == "mujoco_medium" ]]; then
  tasks=( "${mujoco_m_tasks[@]}" )
  env="mujoco"
elif [[ "$environment" == "mujoco_medium_expert" ]]; then
  tasks=( "${mujoco_me_tasks[@]}" )
  env="mujoco"
elif [[ "$environment" == "mujoco_medium_replay" ]]; then
  tasks=( "${mujoco_mr_tasks[@]}" )
  env="mujoco"
elif [[ "$environment" == "antmaze_medium" ]]; then
  tasks=( "${antmaze_m_tasks[@]}" )
  env="antmaze"
elif [[ "$environment" == "antmaze_large" ]]; then
  tasks=( "${antmaze_l_tasks[@]}" )
  env="antmaze"
else
  echo "No valid task provided. Exiting."
  exit 1
fi

# run in a loop
for task in "${tasks[@]}"; do # over tasks in the selected set
  for i in $(shuf -i 0-9999 -n $seeds); do # over random seeds

    echo "Running $task seed $i"
    python dql_d4rl.py \
           env_name="$env" \
           task="$task" \
           loss_type="jacobian" \
           exp_name="jrun_${i}_${task}" \
           seed="$i" \
           project="contractive_diffuser_baseline" \
           wandb_mode="offline" \
           gradient_steps=200000 \
           eval_interval=20000 > "logs/jrun_${i}_${task}.log" 2>&1 &
    done
    wait # wait for all background jobs to finish, place this based on GPU capacity
  done
done

echo "Script finished."
