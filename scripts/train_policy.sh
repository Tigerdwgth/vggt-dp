#!/bin/bash
# Train VGGT-DP on a MetaWorld task.
#
# Usage:
#   bash scripts/train_policy.sh <task_name> <exp_tag> <seed> <gpu_id> [eval_robovis] [robovis] [prio_as_cond] [visual_prio_training]
#
# Example (paper setting):
#   bash scripts/train_policy.sh metaworld_hand-insert vggtdp 0 0 true true true true

set -e

task_name=${1:?task name required, e.g. metaworld_hand-insert}
addition_info=${2:-vggtdp}
seed=${3:-0}
gpu_id=${4:-0}
eval_robovis=${5:-true}
robovis=${6:-true}
policy_prio_as_cond=${7:-true}
policy_visual_prio_training=${8:-true}

alg_name=vggt_dp
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo "gpu id (to use): ${gpu_id}"

export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${gpu_id}
export MUJOCO_GL=${MUJOCO_GL:-egl}

python train.py --config-name=${alg_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=False \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${WANDB_MODE:-online} \
    checkpoint.save_ckpt=True \
    eval_robovis=${eval_robovis} \
    robovis=${robovis} \
    policy.prio_as_cond=${policy_prio_as_cond} \
    policy.visual_prio_training=${policy_visual_prio_training}
