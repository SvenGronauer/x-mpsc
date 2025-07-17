#!/bin/bash
source ../../../.env
task=twolinkarm

cd ${PROJECT_ROOT}/experiments/safe_exp

# #training
python train_safe_exp.py algo=ppo_train_default +task=${task} experiment.tag=train_lr_schedule_longrun