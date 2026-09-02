#!/bin/bash
# One-time environment build. RUN THIS ON THE LOGIN NODE (it needs the proxy),
# not inside a Slurm job:
#     bash cluster/setup_env.sh
#
# Optional:
#     PYTHON_BIN=/usr/bin/python3.11 bash cluster/setup_env.sh
#     bash cluster/setup_env.sh --miniforge     # if no system python 3.10-3.12
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARPO_DIR="${TARPO_DIR:-$(dirname "$HERE")}"
source "$HERE/env.sh"

TORCH_VERSION="2.6.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
XFORMERS_VERSION="0.0.29.post3"

use_conda=0
if [ "${1:-}" = "--miniforge" ] || [ "${1:-}" = "--conda" ]; then use_conda=1; fi

pick_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then echo "$PYTHON_BIN"; return; fi
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys; print("%d%02d" % sys.version_info[:2])')
      if [ "$v" -ge 310 ] && [ "$v" -lt 313 ]; then echo "$c"; return; fi
    fi
  done
  echo ""
}

# Reuse whatever conda you already have under /data/<UPI> (miniconda3,
# miniforge3, anaconda3, ...) instead of installing a second one.
find_conda() {
  local c
  for c in "${DATA_ROOT}"/miniforge3 "${DATA_ROOT}"/miniconda3 "${DATA_ROOT}"/miniconda \
           "${DATA_ROOT}"/anaconda3 "${DATA_ROOT}"/conda "${DATA_ROOT}"/mambaforge; do
    [ -x "$c/bin/conda" ] && { echo "$c"; return; }
  done
  command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null && return
  echo ""
}

# Everything informational goes to stderr here - stdout is the interpreter path.
bootstrap_conda_python() {
  local base env_path="${DATA_ROOT}/envs/tarpo-py"
  base="$(find_conda)"
  if [ -n "$base" ]; then
    echo "==> Reusing the conda install already at $base" >&2
  else
    base="${DATA_ROOT}/miniforge3"
    echo "==> No conda found; installing Miniforge into $base" >&2
    curl -fsSL --proxy "$CLUSTER_PROXY" -o "$TMPDIR/miniforge.sh" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" >&2
    bash "$TMPDIR/miniforge.sh" -b -p "$base" >&2
  fi
  if [ ! -x "$env_path/bin/python" ]; then
    echo "==> Creating conda env $env_path (python 3.11)" >&2
    "$base/bin/conda" create -y -p "$env_path" python=3.11 >&2
  else
    echo "==> Reusing conda env $env_path" >&2
  fi
  echo "$env_path/bin/python"
}

PY="$(pick_python)"
if [ -z "$PY" ] || [ "$use_conda" = "1" ]; then
  [ -z "$PY" ] && echo "==> No system python 3.10-3.12 on PATH; falling back to conda."
  PY="$(bootstrap_conda_python)"
fi
echo "==> Using interpreter: $PY ($("$PY" --version 2>&1))"

if [ -x "$VENV_DIR/bin/python" ]; then
  echo "==> Reusing existing venv at $VENV_DIR (delete it to rebuild from scratch)"
else
  echo "==> Creating venv at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PIP="pip --proxy $CLUSTER_PROXY --timeout 120 --retries 5"
echo "==> Upgrading pip tooling"
$PIP install --upgrade pip wheel setuptools

echo "==> Installing torch ${TORCH_VERSION} (cu124)"
$PIP install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"

echo "==> Installing xformers ${XFORMERS_VERSION}"
$PIP install "xformers==${XFORMERS_VERSION}" --index-url "$TORCH_INDEX" \
  || $PIP install "xformers==${XFORMERS_VERSION}"

echo "==> Installing the rest of the stack"
$PIP install -r "$HERE/requirements-cluster.txt"

echo
echo "==> Environment built. Sanity check (CPU only, no GPU on the login node):"
cd "$TARPO_DIR"
python3 cluster/check_env.py --no-gpu || true

cat <<MSG

Next steps (GSM8K on Qwen2.5-1.5B - see cluster/README.md for MATH):
  1. bash cluster/download_models.sh Qwen/Qwen2.5-1.5B-Instruct
  2. python3 cluster/prepare_data.py --skip-math          # caches gsm8k
  3. echo <your-wandb-api-key> > ${DATA_ROOT}/.wandb_key  # recommended
  4. mkdir -p logs && sbatch cluster/smoke_test.sbatch    # 30-min GPU check
  5. sbatch cluster/train_gsm8k.sbatch
MSG
