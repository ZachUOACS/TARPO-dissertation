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
export WANDB_MODE="${WANDB_MODE:-online}"    # live logging via the squid proxy;
                                             # WANDB_MODE=offline falls back to sync_wandb.sh
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"                  # REQUIRED: puts the vendored transformers/trl/unsloth on sys.path
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
ACTION_BIAS="${ACTION_BIAS:-4.6 0.0}"   # paper: 4.6 0.0 for 1.5B/7B, 2.2 0.0 for 3B

# unsloth writes a generated copy of GRPOTrainer to ./unsloth_compiled_cache and
# never overwrites it (create_new_function(..., overwrite=False)), so any edit to
# the vendored trl/ or transformers/ needs the cache cleared to take effect:
#   RECOMPILE=1 sbatch cluster/train_gsm8k.sbatch
if [ "${RECOMPILE:-0}" = "1" ]; then
  echo "clearing unsloth_compiled_cache"
  rm -rf unsloth_compiled_cache
fi

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
nvidia-smi
python3 cluster/check_env.py || { echo "environment check failed"; exit 1; }

# TARPO paper, Appendix A (Tables 6 and 7), GSM8K column:
#   group size g = 4 (8 is for MATH / DAPO-MATH-17k)
#   total train batch = 32 completions  -> 32 x 1 accumulation
#   soft token top-k = 30               (the script's own default is 10)
#   train lengths = 512 / 512, taken from HRPO Table 5 rather than TARPO's
#     Table 7 (1024/1024), so the run is directly comparable to your HRPO
#     baseline. GSM8K questions are ~100 tokens with ~250-token answers
#     (HRPO Figure 10) and TARPO reports SHORTER completions than baselines,
#     so 512 does not handicap it. To follow TARPO's table instead, pass
#     --max_prompt_length 1024 --max_completion_length 1024 - and re-run HRPO
#     the same way, or the comparison is no longer controlled.
#   action bias b0 = [4.6, 0] for 1.5B and 7B, [2.2, 0] for 3B
# Everything else (lr 5e-6, action-head lr 1e-4, KL beta 0.005, temp 0.5, action
# temp 1.0, LoRA r32/alpha64, cosine + 0.1 warmup, grad norm 0.1, AdamW-8bit,
# 1 epoch) already matches the script defaults.
#
# One epoch = ceil(7473 * 4 / 32) = 935 optimizer steps (934 full batches plus a
# partial one) - the same step count as HRPO. Checkpoints land every 250 steps
# and the Trainer writes a final checkpoint-935 when training stops.
#
# HRPO Table 4 reaches its total of 32 as per_device 8 x accumulation 4. That is
# the SAME optimizer step as 32 x 1 used here (identical 32 completions, and
# advantages are standardized within each 4-completion group, not per micro-
# batch), so it is a memory knob, not a comparability one - 32 x 1 just samples
# them in one generate() call instead of four and is roughly 2x faster on one
# GPU. If it OOMs, hold the total at 32 as you step down:
#   --per_device_train_batch_size 16 --gradient_accumulation_steps 2
#   --per_device_train_batch_size 8  --gradient_accumulation_steps 4  (= HRPO)
# Per-step wall clock is then not comparable to your HRPO run; compare the
# logged completion_length and reward instead, which is what TARPO claims on.
python3 tarpo_gsm8k.py \
  --model_name "${MODEL_ROOT}/${MODEL_NAME}" \
  --group_size 4 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --soft_top_k 30 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --action_bias ${ACTION_BIAS} \
  "$@"

echo "finished ($(date))"
# If the wall clock killed the run instead, just submit this script again:
# tarpo_gsm8k.py picks up the newest checkpoint-* automatically (--resume never to opt out).
