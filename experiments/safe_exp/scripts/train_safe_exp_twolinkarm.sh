#!/bin/bash
source "../../../.env"
cd ${PROJECT_ROOT}/experiments/safe_exp

# general settings
task="twolinkarm"
OUTPUT_DIR=${DATA_DIR}/outputs/TwoLinkArm-v0/safe_explorer_ppo/pretrain/$(date +"%Y-%m-%d_%H-%M-%S")
MODEL_PATH=${OUTPUT_DIR}/checkpoints/model_latest.pt
# MODEL_PATH=${DATA_DIR}/outputs/TwoLinkArm-v0/safe_explorer_ppo/pretrain/2023-01-23_13-25-47/checkpoints/model_latest.pt

# # pretraining
python train_safe_exp.py +algo=safe_exp_ppo_pretrain +task=${task} experiment.tag=pretrain hydra.run.dir=$OUTPUT_DIR


# #trainings
slacks=(0.05 0.10 0.15 0.2)
seeds=(42 43 44)
for slack in "${slacks[@]}"
do
    for seed in "${seeds[@]}"
    do
        python train_safe_exp.py +algo=safe_exp_ppo_train +task=${task} algo.config.pretrained=$MODEL_PATH algo.config.constraint_slack=${slack} experiment.tag=algo_constraint_slack_${slack} experiment.seed=$seed &
    done
done
wait