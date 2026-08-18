#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=tarpo-smoke
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/tarpo-smoke-%j.out.log
#SBATCH --error=logs/tarpo-smoke-%j.err.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 30-minute check that the whole stack works on a GPU before you queue a real
# run: imports, CUDA, bitsandbytes, the vendored ActionHead, and ~10 minutes of
# actual GRPO steps on GSM8K with a tiny configuration.
#     mkdir -p logs && sbatch cluster/smoke_test.sbatch
set -euo pipefail

export TARPO_DIR="${TARPO_DIR:-/data/${USER}/TARPO-dissertation}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
nvidia-smi
python3 cluster/check_env.py --model "${MODEL_ROOT}/${MODEL_NAME}"

# Deliberately odd hyperparameters so this lands in its own ./experiments
# directory and never collides with a real run.
echo
echo "=== short GSM8K training run (10 min, then killed on purpose) ==="
timeout 600 python3 tarpo_gsm8k.py \
  --model_name "${MODEL_ROOT}/${MODEL_NAME}" \
  --group_size 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_prompt_length 256 \
  --max_completion_length 256 \
  --lora_rank 8 \
  --action_bias 4.6 0.0 || echo "(stopped after the timeout - that is expected)"

echo
echo "smoke test done ($(date))."
echo "What to look for above: 'All checks passed', the ActionHead custom_init line,"
echo "and at least one 'Batch accuracy: YN...' block from the reward function."
echo "Then delete the smoke-test experiment dir (the one with -lora8-)."
