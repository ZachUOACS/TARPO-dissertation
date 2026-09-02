# shellcheck shell=bash
# Common environment for the UoA ML cluster (foscsmlprd0{1,2,3}).
# Source this from login-node helpers AND from every Slurm job:
#     source /data/$USER/TARPO-dissertation/cluster/env.sh
#
# Everything lives under /data/<UPI> per the cluster guidelines - nothing is
# written to /home (files outside /data/<UPI> get deleted without notice).

# --- Identity / paths -------------------------------------------------------
UPI="${UPI:-$USER}"
export DATA_ROOT="/data/${UPI}"
export HOME="${DATA_ROOT}"                      # required by the guidelines
export TMPDIR="${DATA_ROOT}/tmp"                # required by the guidelines
export TARPO_DIR="${TARPO_DIR:-${DATA_ROOT}/TARPO-dissertation}"
export VENV_DIR="${VENV_DIR:-${DATA_ROOT}/envs/tarpo}"
export MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/models}"

# --- Outbound internet (login node only; compute nodes are usually walled) ---
export CLUSTER_PROXY="${CLUSTER_PROXY:-http://squid.auckland.ac.nz:3128}"
if [ "${USE_PROXY:-1}" = "1" ]; then
  export HTTP_PROXY="$CLUSTER_PROXY"  http_proxy="$CLUSTER_PROXY"
  export HTTPS_PROXY="$CLUSTER_PROXY" https_proxy="$CLUSTER_PROXY"
  export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
fi

# --- Hugging Face -----------------------------------------------------------
export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=0   # unsloth turns this on; the pkg isn't installed
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # sbatch scripts set this to 1
export TOKENIZERS_PARALLELISM=false

# --- unsloth ----------------------------------------------------------------
export UNSLOTH_DISABLE_AUTO_UPDATES=1   # stops unsloth pip-installing at import
export PYTHONNOUSERSITE=1
# Belt and braces: python only puts the *script's* directory on sys.path, so a
# helper living in cluster/ would miss the vendored transformers/trl/unsloth in
# the repo root. This makes them win from anywhere.
export PYTHONPATH="${TARPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# --- Weights & Biases -------------------------------------------------------
# The training scripts hardcode  os.environ["WANDB_PROJECT"] = "latent-reasoning"
# and report_to="wandb", so wandb must be importable and usable.
export WANDB_DIR="${DATA_ROOT}/wandb"
export WANDB_CACHE_DIR="${DATA_ROOT}/.cache/wandb"
export WANDB_DATA_DIR="${DATA_ROOT}/.cache/wandb-data"
export WANDB_MODE="${WANDB_MODE:-online}"    # compute nodes reach wandb.ai via the proxy
                                             # (verified job 15067); WANDB_MODE=offline +
                                             # cluster/sync_wandb.sh is the fallback
[ -f "${DATA_ROOT}/.wandb_key" ] && export WANDB_API_KEY="$(cat "${DATA_ROOT}/.wandb_key")"
# export WANDB_ENTITY=your-wandb-username-or-team   # optional, put it in .wandb_entity
[ -f "${DATA_ROOT}/.wandb_entity" ] && export WANDB_ENTITY="$(cat "${DATA_ROOT}/.wandb_entity")"

# --- Threads ----------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$TMPDIR" "$HF_HOME" "$MODEL_ROOT" "$WANDB_DIR" "$(dirname "$VENV_DIR")"
