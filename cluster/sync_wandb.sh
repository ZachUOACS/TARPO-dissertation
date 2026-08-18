#!/bin/bash
# Push offline W&B runs to wandb.ai. RUN ON THE LOGIN NODE (needs the proxy).
#     bash cluster/sync_wandb.sh            # sync everything not yet synced
#     bash cluster/sync_wandb.sh <run-dir>  # sync one run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARPO_DIR="${TARPO_DIR:-$(dirname "$HERE")}"
source "$HERE/env.sh"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ ! -f "${DATA_ROOT}/.wandb_key" ]; then
  echo "No API key at ${DATA_ROOT}/.wandb_key."
  echo "Get it from https://wandb.ai/authorize, then:"
  echo "  echo <key> > ${DATA_ROOT}/.wandb_key && chmod 600 ${DATA_ROOT}/.wandb_key"
  exit 1
fi
export WANDB_MODE=online
wandb login --relogin "$(cat "${DATA_ROOT}/.wandb_key")"

if [ $# -gt 0 ]; then
  wandb sync "$@"
else
  # offline runs land in $WANDB_DIR/wandb/offline-run-*
  cd "$WANDB_DIR"
  wandb sync --sync-all
fi
