#!/usr/bin/env bash
set -euo pipefail
: "${TARGET:?Set TARGET to vicuna-13b-v1.3}"
: "${DRAFT:?Set DRAFT to its matching EAGLE-3 checkpoint}"
: "${QUESTIONS:?Set QUESTIONS to the question JSONL}"
: "${OUT:?Set OUT to a new result directory}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(realpath "$TARGET")"; DRAFT="$(realpath "$DRAFT")"
QUESTIONS="$(realpath "$QUESTIONS")"; OUT="$(realpath -m "$OUT")"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
exec "${PYTHON:-python}" run_pair.py --model-profile vicuna13b --target "$TARGET" --draft "$DRAFT" --questions "$QUESTIONS" --output "$OUT" "$@"
