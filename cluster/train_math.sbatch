#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=tarpo-math
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/tarpo-math-%j.out.log
#SBATCH --error=logs/tarpo-math-%j.err.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Submit from the repo root:  mkdir -p logs && sbatch cluster/train_math.sbatch
# Override anything on the command line, e.g.
#   MODEL_NAME=Qwen2.5-1.5B-Instruct sbatch cluster/train_math.sbatch --action_bias 4.6 0.0
set -euo pipefail

export TARPO_DIR="${TARPO_DIR:-/data/${USER}/TARPO-dissertation}"
export HF_HUB_OFFLINE=1          # compute nodes have no outbound internet
export WANDB_MODE="${WANDB_MODE:-offline}"   # sync later: bash cluster/sync_wandb.sh
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"                  # REQUIRED: puts the vendored transformers/trl/unsloth on sys.path
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
ACTION_BIAS="${ACTION_BIAS:-2.2 0.0}"   # paper: 4.6 0.0 for 1.5B/7B, 2.2 0.0 for 3B

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
nvidia-smi
python3 cluster/check_env.py || { echo "environment check failed"; exit 1; }

# per_device_train_batch_size must be a multiple of group_size (TRL requirement).
# 8 x 8 accumulation = the paper's effective batch of 64 completions, but fits
# one GPU. Raise per_device_train_batch_size (16, 32, 64) if memory allows.
python3 tarpo_math.py \
  --model_name "${MODEL_ROOT}/${MODEL_NAME}" \
  --dataset_root data/MATH \
  --group_size 8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --action_bias ${ACTION_BIAS} \
  "$@"

echo "finished ($(date))"
# If the wall clock killed the run instead, just submit this script again:
# tarpo_math.py picks up the newest checkpoint-* automatically (--resume never to opt out).
