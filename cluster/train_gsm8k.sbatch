#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=tarpo-gsm8k
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/tarpo-gsm8k-%j.out.log
#SBATCH --error=logs/tarpo-gsm8k-%j.err.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# TARPO on GSM8K with Qwen2.5-1.5B-Instruct - the cheapest way into this repo:
# no local dataset to build (gsm8k comes from the HF cache) and the smallest
# backbone in the paper.
#
#   mkdir -p logs && sbatch cluster/train_gsm8k.sbatch
#
# Override on the command line, e.g.
#   MODEL_NAME=Qwen2.5-3B-Instruct ACTION_BIAS="2.2 0.0" sbatch cluster/train_gsm8k.sbatch
#   sbatch cluster/train_gsm8k.sbatch --max_completion_length 1024   # paper-length
set -euo pipefail

export TARPO_DIR="${TARPO_DIR:-/data/${USER}/TARPO-dissertation}"
export HF_HUB_OFFLINE=1          # gsm8k must already be in the HF cache
                                 # (cluster/prepare_data.py does that on the login node)
export WANDB_MODE="${WANDB_MODE:-offline}"   # sync later: bash cluster/sync_wandb.sh
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"                  # REQUIRED: puts the vendored transformers/trl/unsloth on sys.path
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
ACTION_BIAS="${ACTION_BIAS:-4.6 0.0}"   # paper: 4.6 0.0 for 1.5B/7B, 2.2 0.0 for 3B

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
nvidia-smi
python3 cluster/check_env.py || { echo "environment check failed"; exit 1; }

# per_device_train_batch_size must be a multiple of group_size (TRL requirement);
# 8 x 8 accumulation gives the paper's effective batch of 64 completions on one GPU.
#
# Lengths: GSM8K questions are ~100 tokens and its CoT answers a few hundred, so
# 512/512 costs nothing in truncation and roughly halves generation time versus
# the 1024/1024 the MATH runs use. Generation is the bottleneck here - vLLM is
# off (use_vllm=False), so every completion is sampled with plain HF generate.
python3 tarpo_gsm8k.py \
  --model_name "${MODEL_ROOT}/${MODEL_NAME}" \
  --group_size 8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --action_bias ${ACTION_BIAS} \
  "$@"

echo "finished ($(date))"
# If the wall clock killed the run instead, just submit this script again:
# tarpo_gsm8k.py picks up the newest checkpoint-* automatically (--resume never to opt out).
