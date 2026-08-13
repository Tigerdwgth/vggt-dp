#!/bin/bash
# Generate expert demonstrations for a MetaWorld task using the scripted expert policy.
#
# Usage:
#   bash scripts/gen_demonstration_metaworld.sh <task_name> [robovis] [num_episodes]
#
# Example:
#   bash scripts/gen_demonstration_metaworld.sh hand-insert True 10
#
# Output: data/metaworld_<task_name>_expert.zarr
#         (robovis=False writes data/metaworld_<task_name>_expert_invis.zarr)

set -e

task_name=${1:?task name required, e.g. hand-insert}
robovis=${2:-True}
num_episodes=${3:-10}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}

cd "${repo_root}/third_party/Metaworld"
python gen_demonstration_expert.py --env_name=${task_name} \
    --num_episodes ${num_episodes} \
    --root_dir "${repo_root}/data/" \
    --robovis=${robovis}
