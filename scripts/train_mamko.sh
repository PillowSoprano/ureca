#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/train_mamko.sh [standard|hybrid]

Run MamKO training for the wastewater task.

Arguments:
  standard   Train on simulated wastewater data (default).
             Honors the following environment variables:
               EPOCHS      - number of epochs to train (default: 401)
               BATCH_SIZE  - batch size (default: 256)

  hybrid     Train in hybrid mode using real + simulated data.

Examples:
  scripts/train_mamko.sh
  EPOCHS=500 BATCH_SIZE=128 scripts/train_mamko.sh standard
  scripts/train_mamko.sh hybrid
USAGE
}

MODE="${1:-standard}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  standard)
    EPOCHS="${EPOCHS:-401}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    CMD=(python train.py mamba wastewater --epochs "$EPOCHS" --batch_size "$BATCH_SIZE")
    ;;
  hybrid)
    CMD=(python train.py mamba wastewater --training_mode hybrid)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo
    usage >&2
    exit 1
    ;;
esac

echo "Running: ${CMD[*]}"
"${CMD[@]}"
