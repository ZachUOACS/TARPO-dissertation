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
export WANDB_MODE="${WANDB_MODE:-online}"    # live logging via the squid proxy;
                                             # WANDB_MODE=offline falls back to sync_wandb.sh
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"                  # REQUIRED: puts the vendored transformers/trl/unsloth on sys.path
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
ACTION_BIAS="${ACTION_BIAS:-2.2 0.0}"   # paper: 4.6 0.0 for 1.5B/7B, 2.2 0.0 for 3B

# unsloth writes a generated copy of GRPOTrainer to ./unsloth_compiled_cache and
# never overwrites it, so any edit to the vendored trl/ or transformers/ needs
# the cache cleared:  RECOMPILE=1 sbatch cluster/train_math.sbatch
if [ "${RECOMPILE:-0}" = "1" ]; then
  echo "clearing unsloth_compiled_cache"
  rm -rf unsloth_compiled_cache
fi

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
nvidia-smi
python3 cluster/check_env.py || { echo "environment check failed"; exit 1; }

# MATH column of TARPO Table 6/7, which HRPO Table 4/5 matches except for the
# prompt budget (HRPO 512, TARPO 1024 - TARPO's is used here, it only widens the
# truncation margin):
#   group size g = 8, total train batch = 64 completions
#   soft token top-k = 30, lengths 1024 / 1024
#   action bias b0 = [2.2, 0] for 3B, [4.6, 0] for 1.5B and 7B
# HRPO reaches its 64 as 16 x accumulation 4; 64 x 1 here is the same optimizer
# step (advantages are standardized per 8-completion group either way) but
# samples it in one generate() call. If it OOMs, hold the total at 64:
# 32 x 2, then 16 x 4 (= HRPO's layout).
python3 tarpo_math.py \
  --model_name "${MODEL_ROOT}/${MODEL_NAME}" \
  --dataset_root data/MATH \
  --group_size 8 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --soft_top_k 30 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --action_bias ${ACTION_BIAS} \
  "$@"

echo "finished ($(date))"
# If the wall clock killed the run instead, just submit this script again:
# tarpo_math.py picks up the newest checkpoint-* automatically (--resume never to opt out).
