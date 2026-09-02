#!/bin/bash
# rsync this working copy to /data/<UPI>/TARPO-dissertation on the cluster, and
# pull results back. Run it from the repo root on your own machine.
#
#   export CLUSTER_UPI=ztay632            # once per shell (or put it in ~/.bashrc)
#   bash cluster/sync_to_cluster.sh push --dry-run
#   bash cluster/sync_to_cluster.sh push
#   bash cluster/sync_to_cluster.sh pull            # eval results + logs (small)
#   bash cluster/sync_to_cluster.sh pull-all        # whole experiments/ dir
#   CLUSTER_HOST=foscsmlprd02.its.auckland.ac.nz bash cluster/sync_to_cluster.sh push
#
# Password is <upi_password>:<2FA_code>. The first connection opens a shared
# SSH control socket that later rsyncs reuse, so you only type it once per
# 10 minutes instead of once per command.
set -euo pipefail

: "${CLUSTER_UPI:?set CLUSTER_UPI=<your UPI>, e.g. export CLUSTER_UPI=ztay632}"
CLUSTER_HOST="${CLUSTER_HOST:-foscsmlprd01.its.auckland.ac.nz}"
REMOTE_DIR="${REMOTE_DIR:-/data/${CLUSTER_UPI}/TARPO-dissertation}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${CLUSTER_UPI}@${CLUSTER_HOST}"

CTL="${HOME}/.ssh/cm-%r@%h:%p"
mkdir -p "${HOME}/.ssh"
SSH="ssh -o ControlMaster=auto -o ControlPath=${CTL} -o ControlPersist=10m"

# Never send these; because they are filtered on the sender they are also
# protected from --delete on the receiver, so cluster-side checkpoints, data and
# model symlinks survive a push.
EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '/unsloth_compiled_cache/'   # regenerated on the cluster from source
  --exclude '.venv/' --exclude 'venv/' --exclude 'envs/'
  --exclude '/experiments/'      # checkpoints live only on the cluster
  --exclude '/data/'             # datasets are built on the cluster
  --exclude '/logs/'
  --exclude '/wandb/'
  --exclude '/models/'
  --exclude '/Qwen2.5-*'         # symlinks created by download_models.sh
  --exclude '/Llama-3.1-*'
  --exclude '*.log'
  --exclude '.DS_Store'
)

MODE="${1:-push}"; shift || true

case "$MODE" in
  push)
    echo "==> ${LOCAL_DIR}/  ->  ${TARGET}:${REMOTE_DIR}/"
    $SSH "$TARGET" "mkdir -p '${REMOTE_DIR}'"
    rsync -avz --human-readable --partial --progress \
      -e "$SSH" "${EXCLUDES[@]}" "$@" \
      "${LOCAL_DIR}/" "${TARGET}:${REMOTE_DIR}/"
    ;;
  pull)
    # metrics only: eval json/summaries and job logs, no model weights
    echo "==> ${TARGET}:${REMOTE_DIR}  ->  ${LOCAL_DIR}  (results only)"
    rsync -avz --human-readable --partial --progress -e "$SSH" \
      --include '*/' \
      --include 'eval_*/**' --include '*.json' --include '*.jsonl' --include '*.log' \
      --exclude '*' "$@" \
      "${TARGET}:${REMOTE_DIR}/experiments/" "${LOCAL_DIR}/experiments/"
    rsync -avz --human-readable -e "$SSH" "$@" \
      "${TARGET}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/logs/" || true
    ;;
  pull-all)
    echo "==> ${TARGET}:${REMOTE_DIR}/experiments  ->  ${LOCAL_DIR}/experiments (everything)"
    rsync -avz --human-readable --partial --progress -e "$SSH" "$@" \
      "${TARGET}:${REMOTE_DIR}/experiments/" "${LOCAL_DIR}/experiments/"
    ;;
  shell)
    exec $SSH "$TARGET"
    ;;
  *)
    echo "usage: $0 {push|pull|pull-all|shell} [extra rsync flags]" >&2
    exit 2
    ;;
esac
