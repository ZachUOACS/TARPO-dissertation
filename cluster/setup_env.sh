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

use_miniforge=0
[ "${1:-}" = "--miniforge" ] && use_miniforge=1

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

install_miniforge() {
  local mf="${DATA_ROOT}/miniforge3"
  if [ ! -x "$mf/bin/python" ]; then
    echo "==> Installing Miniforge into $mf"
    curl -fsSL --proxy "$CLUSTER_PROXY" -o "$TMPDIR/miniforge.sh" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$TMPDIR/miniforge.sh" -b -p "$mf"
  fi
  "$mf/bin/conda" create -y -p "${DATA_ROOT}/envs/tarpo-py" python=3.11 >/dev/null
  echo "${DATA_ROOT}/envs/tarpo-py/bin/python"
}

PY="$(pick_python)"
if [ -z "$PY" ] || [ "$use_miniforge" = "1" ]; then
  echo "==> No suitable system python 3.10-3.12 found (or --miniforge given)."
  PY="$(install_miniforge)"
fi
echo "==> Using interpreter: $PY ($("$PY" --version 2>&1))"

echo "==> Creating venv at $VENV_DIR"
"$PY" -m venv "$VENV_DIR"
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

Next steps:
  1. bash cluster/download_models.sh Qwen/Qwen2.5-3B-Instruct
  2. python3 cluster/prepare_data.py            # MATH + eval datasets (login node)
  3. echo <your-wandb-api-key> > ${DATA_ROOT}/.wandb_key   # optional but recommended
  4. mkdir -p logs && sbatch cluster/smoke_test.sbatch     # 15-min GPU check
  5. sbatch cluster/train_math.sbatch
MSG
