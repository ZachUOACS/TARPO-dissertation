#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=tarpo-eval
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/tarpo-eval-%j.out.log
#SBATCH --error=logs/tarpo-eval-%j.err.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Evaluate a trained checkpoint. CKPT must keep the base-model name and the
# -temp.../-topk... tags in its path: the eval scripts parse them back out of it.
#
#   CKPT=experiments/Qwen2.5-1.5B-Instruct-gsm8k-tarpo-.../checkpoint-500 \
#     sbatch cluster/eval.sbatch
#
#   EVAL_SCRIPT=eval_tarpo_math_avg.py CKPT=... sbatch cluster/eval.sbatch
#   EVAL_SCRIPT=eval_tarpo_amc_avg.py DATA_PATH=data/amc23.json CKPT=... sbatch cluster/eval.sbatch
#
# k=32 samples per question is the paper's Pass@32 setting and is slow; K=8 or
# K=4 is plenty for a first sanity check of a run.
set -euo pipefail

export TARPO_DIR="${TARPO_DIR:-/data/${USER}/TARPO-dissertation}"
export HF_HUB_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-online}"
source "${TARPO_DIR}/cluster/env.sh"
cd "$TARPO_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

: "${CKPT:?set CKPT=<path to a checkpoint-* directory>}"
SCRIPT="${EVAL_SCRIPT:-eval_tarpo_gsm8k_avg.py}"
K="${K:-8}"
BS="${BS:-4}"

# The eval scripts call from_pretrained("Qwen2.5-1.5B-Instruct") with no org
# prefix, so that bare name must exist as a directory/symlink in the repo root.
BASE="$(python3 - "$CKPT" <<'PY'
import sys
for m in ("Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"):
    if m in sys.argv[1]:
        print(m); break
PY
)"
if [ -z "$BASE" ] || [ ! -e "$BASE" ]; then
  echo "Base model '${BASE:-<none found in CKPT path>}' is not present in the repo root."
  echo "Run: bash cluster/download_models.sh Qwen/${BASE:-Qwen2.5-1.5B-Instruct}"
  exit 1
fi

ARGS=(--checkpoint_path "$CKPT" --batch_size "$BS" --k "$K")
case "$SCRIPT" in
  eval_tarpo_gsm8k_avg.py)
    # --data_path is required by its argparse but never read: the script pulls
    # gsm8k straight from the HF cache. Any string satisfies it.
    ARGS+=(--data_path "${DATA_PATH:-unused}") ;;
  eval_tarpo_math_avg.py)
    : ;;  # MATH-500 comes from the HF cache
  *)
    : "${DATA_PATH:?$SCRIPT needs DATA_PATH=<benchmark>.json}"
    ARGS+=(--data_path "$DATA_PATH") ;;
esac

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  ($(date))"
echo "eval: $SCRIPT  base: $BASE  k=$K  batch=$BS"
nvidia-smi
python3 "$SCRIPT" "${ARGS[@]}" "$@"
echo "results written under $CKPT ($(date))"
