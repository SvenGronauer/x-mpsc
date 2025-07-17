#!/bin/bash
source "../../../.env"
cd ${PROJECT_ROOT}/experiments/safe_exp

# general settings
task="hopper"
OUTPUT_DIR=${DATA_DIR}/outputs/SafeHopper-v4/safe_explorer_ppo/pretrain/$(date +"%Y-%m-%d_%H-%M-%S")
MODEL_PATH=${OUTPUT_DIR}/checkpoints/model_latest.pt

# # pretraining
python train_safe_exp.py +algo=safe_exp_ppo_pretrain +task=${task} experiment.tag=pretrain hydra.run.dir=$OUTPUT_DIR

# #training
slacks=(0.05 0.10 0.15)
seeds=(42 43 44)
for slack in "${slacks[@]}"
do
    for seed in "${seeds[@]}"
    do
        python train_safe_exp.py +algo=safe_exp_ppo_train +task=${task} algo.config.pretrained=$MODEL_PATH algo.config.constraint_slack=${slack} experiment.tag=algo_constraint_slack_${slack} experiment.seed=$seed &
    done
done
wait