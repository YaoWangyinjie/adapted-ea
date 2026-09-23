#!/usr/bin/env bash
set -euo pipefail
: "${TARGET:?Set TARGET to the local target model directory}"
: "${DRAFT:?Set DRAFT to its matching EAGLE-3 checkpoint}"
: "${QUESTIONS:?Set QUESTIONS to the dataset JSONL}"
: "${OUT:?Set OUT to a new result directory}"
PROFILE="${PROFILE:-deepseek}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PYTHON="${PYTHON:-python}"
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo 'Use one GPU index, e.g. GPU=0 or CUDA_VISIBLE_DEVICES=0.' >&2
  exit 2
fi
# Resolve paths before switching to the repository/package directory.
TARGET="$(realpath "$TARGET")"
DRAFT="$(realpath "$DRAFT")"
QUESTIONS="$(realpath "$QUESTIONS")"
OUT="$(realpath -m "$OUT")"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$PYTHON" -u run_comparison.py "$@" \
  --model-profile "$PROFILE" --target "$TARGET" --draft "$DRAFT" \
  --questions "$QUESTIONS" --output "$OUT"
