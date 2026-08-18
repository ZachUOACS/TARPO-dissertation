#!/bin/bash
# Download a base model into /data/<UPI>/models and symlink it into the repo
# root. RUN ON THE LOGIN NODE (needs the proxy).
#
#     bash cluster/download_models.sh Qwen/Qwen2.5-3B-Instruct
#     bash cluster/download_models.sh Qwen/Qwen2.5-1.5B-Instruct
#
# The symlink matters: the eval scripts infer the base model from the checkpoint
# path and then call from_pretrained("Qwen2.5-3B-Instruct") with no org prefix,
# so that name has to resolve as a directory in the repo root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARPO_DIR="${TARPO_DIR:-$(dirname "$HERE")}"
source "$HERE/env.sh"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

REPO_ID="${1:-Qwen/Qwen2.5-3B-Instruct}"
NAME="$(basename "$REPO_ID")"
DEST="${MODEL_ROOT}/${NAME}"

# Gated repos (Llama) need a token: echo <hf_token> > $DATA_ROOT/.hf_token
[ -f "${DATA_ROOT}/.hf_token" ] && export HF_TOKEN="$(cat "${DATA_ROOT}/.hf_token")"

echo "==> Downloading ${REPO_ID} -> ${DEST}"
python3 - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id,
    local_dir=dest,
    ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*", "*.gguf"],
    max_workers=4,
)
print("saved to", path)
PY

ln -sfn "$DEST" "${TARPO_DIR}/${NAME}"
echo "==> Symlinked ${TARPO_DIR}/${NAME} -> ${DEST}"
echo "    train with:  --model_name ${MODEL_ROOT}/${NAME}"
