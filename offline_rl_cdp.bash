#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./all_exps.bash <script> <seeds>

set -euo pipefail

# params
environment="${1:-"kitchen"}"
seeds="${2:-1}"

# create logs directory if it doesn't exist
mkdir -p logs

# main script logic goes here
echo "Starting script with argument: $environment, $seeds"

# list of tasks, devided by environment type and expert
antmaze_m_tasks=("antmaze-medium-play-v2" "antmaze-medium-diverse-v2")
antmaze_l_tasks=("antmaze-large-play-v2" "antmaze-large-diverse-v2")
kitchen_tasks=("kitchen-mixed-v0" "kitchen-partial-v0" "kitchen-complete-v0")
mujoco_m_tasks=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2")
mujoco_me_tasks=("hopper-medium-expert-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-expert-v2")
mujoco_mr_tasks=("halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2")

# declare associative arrays for tasks and environments
declare -A task_map env_map
task_map=(
  ["kitchen"]="${kitchen_tasks[*]}"
  ["mujoco_medium"]="${mujoco_m_tasks[*]}"
  ["mujoco_medium_expert"]="${mujoco_me_tasks[*]}"
  ["mujoco_medium_replay"]="${mujoco_mr_tasks[*]}"
  ["antmaze_medium"]="${antmaze_m_tasks[*]}"
  ["antmaze_large"]="${antmaze_l_tasks[*]}"
)

env_map=(
  ["kitchen"]="kitchen"
  ["mujoco_medium"]="mujoco"
  ["mujoco_medium_expert"]="mujoco"
  ["mujoco_medium_replay"]="mujoco"
  ["antmaze_medium"]="antmaze"
  ["antmaze_large"]="antmaze"
)

# check if environment is valid and set tasks/env
if [[ -v task_map["$environment"] ]]; then
  read -ra tasks <<< "${task_map["$environment"]}"
  env="${env_map["$environment"]}"
else
  echo "No valid task provided. Exiting."
  exit 1
fi

# configuration
loss_weights=(0.0 0.001) # loss coefficients to sweep over
gradient_steps=500000
eval_interval=50000
wandb_mode="offline"
loss_type="jacobian"

# run experiments with nested loops
for penalty in "${loss_weights[@]}"; do
  echo "Starting experiments with loss weight: $penalty"

  for task in "${tasks[@]}"; do
    echo "  Running task: $task"

    # generate random seeds and run experiments
    for seed in $(shuf -i 0-9999 -n "$seeds"); do
      exp_name="jrun_${seed}_${task}_${penalty}"
      log_file="logs/${exp_name}.log"

      echo "    Launching seed $seed (log: $log_file)"

      python scripts/cdp_rl.py \
        env_name="$env" \
        task="$task" \
        loss_type="$loss_type" \
        exp_name="$exp_name" \
        seed="$seed" \
        project="contractive_diffuser" \
        gradient_steps="$gradient_steps" \
        eval_interval="$eval_interval" \
        wandb_mode="$wandb_mode" \
        loss_weights.jacobian="$penalty" \
        > "$log_file" 2>&1 &
    done

    # wait for all tasks of current penalty/task combination to complete
    wait
    echo "  Completed all seeds for task: $task"
  done

  echo "Completed all tasks for jacobian penalty: $penalty"
done

echo "Script finished."
