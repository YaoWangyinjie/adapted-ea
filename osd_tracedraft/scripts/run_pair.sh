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
# These arguments are controlled by the script, not forwarded overrides.
for arg in "$@"; do
  case "$arg" in
    --method|--method=*|--output|--output=*|--config|--config=*)
      echo "Set paths through environment variables; do not pass $arg here." >&2
      exit 2 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for method in osd osd_tracedraft; do
  "$PYTHON" -u -m osd_tracedraft.run "$@" \
    --model-profile "$PROFILE" --method "$method" \
    --target "$TARGET" --draft "$DRAFT" --questions "$QUESTIONS" \
    --output "$OUT/$method"
done
"$PYTHON" -m osd_tracedraft.report \
  --runs "$OUT/osd" "$OUT/osd_tracedraft" \
  --reference-method osd --output "$OUT/comparison"
