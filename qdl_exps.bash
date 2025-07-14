#!/usr/bin/env bash

# description: Run batched experiments with a given script and number of seeds.
# usage: ./all_exps.bash <script> <seeds>

set -euo pipefail
seeds=3

# for ((i=0; i<seeds; i++)); do
#   echo "Running plain run for seed $i on kitchen:kitchen-mixed-v0"
#   python contractive_dql.py exp_name=plain_run_${i}_0_kitchen-mixed-v0 gradient_steps=400000 diffusion_steps=50  sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False seed=$i > logs/output/plain_run_${i}_0_kitchen-mixed-v0.log 2>&1 &
# done

solver=ddim
predict_noise=False
gradient_steps=550000
diffusion_steps=10
sampling_steps=10
jacobian_weight=(0.001 -0.005 -0.01)

# for ((i=0; i<seeds; i++)); do
#   echo "Running jacobian run for seed $i on kitchen:kitchen-mixed-v0"
#   python contractive_dql.py exp_name=jacobian_run_${i}_5_kitchen-mixed-v0 gradient_steps=$gradient_steps loss_weights.jacobian=${jacobian_weight[0]} diffusion_steps=$diffusion_steps sampling_steps=$sampling_steps solver=$solver predict_noise=$predict_noise seed=$i > logs/output/jacobian_run_${i}_5_kitchen-mixed-v0.log 2>&1 &
# done

# wait

for ((i=0; i<seeds; i++)); do
  echo "Running jacobian run for seed $i on kitchen:kitchen-mixed-v0"
  python contractive_dql.py exp_name=jacobian_run_${i}_${jacobian_weight[1]}_kitchen-mixed-v0 gradient_steps=$gradient_steps loss_weights.jacobian=${jacobian_weight[1]} diffusion_steps=$diffusion_steps  sampling_steps=$sampling_steps solver=$solver predict_noise=$predict_noise seed=$i > logs/output/jacobian_run_${i}_${jacobian_weight[1]}_kitchen-mixed-v0.log 2>&1 &
done

wait

for ((i=0; i<seeds; i++)); do
  echo "Running jacobian run for seed $i on kitchen:kitchen-mixed-v0"
  python contractive_dql.py exp_name=jacobian_run_${i}_${jacobian_weight[2]}_kitchen-mixed-v0 gradient_steps=$gradient_steps loss_weights.jacobian=${jacobian_weight[2]} diffusion_steps=$diffusion_steps  sampling_steps=$sampling_steps solver=$solver predict_noise=$predict_noise seed=$i > logs/output/jacobian_run_${i}_${jacobian_weight[2]}_kitchen-mixed-v0.log 2>&1 &
done

# python contractive_dql.py exp_name=jacobina_run_1_2_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=2 diffusion_steps=50  sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_2_kitchen-mixed-v0.log 2>&1 &

# python contractive_dql.py exp_name=jacobina_run_1_4_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=4 diffusion_steps=50 sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_4_kitchen-mixed-v0.log 2>&1 &

# python contractive_dql.py exp_name=jacobina_run_1_8_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=8 diffusion_steps=50 sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_8_kitchen-mixed-v0.log 2>&1 &

# python contractive_dql.py exp_name=jacobina_run_1_12_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=12 diffusion_steps=50 sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_12_kitchen-mixed-v0.log 2>&1 &

# python contractive_dql.py exp_name=jacobina_run_1_16_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=16 diffusion_steps=50 sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_16_kitchen-mixed-v0.log 2>&1 &

# python contractive_dql.py exp_name=jacobina_run_1_32_kitchen-mixed-v0 gradient_steps=1000000 loss_weights.jacobian=32 diffusion_steps=50 sampling_steps=15 solver=ode_dpmsolver++_2M predict_noise=False > logs/output/jacobina_run_1_32_kitchen-mixed-v0.log 2>&1 &

echo "Script finished."