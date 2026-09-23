#!/usr/bin/env bash
set -euo pipefail
: "${TARGET:?Set TARGET to the downloaded target directory}"
: "${DRAFT:?Set DRAFT to the matching EAGLE-3 checkpoint}"
: "${QUESTIONS:?Set QUESTIONS to the question JSONL}"
: "${OUT:?Set OUT to a new result directory}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve relative user paths before changing the working directory.
TARGET="$(realpath "$TARGET")"; DRAFT="$(realpath "$DRAFT")"
QUESTIONS="$(realpath "$QUESTIONS")"; OUT="$(realpath -m "$OUT")"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
exec "${PYTHON:-python}" run_pair.py --model-profile deepseek --target "$TARGET" --draft "$DRAFT" --questions "$QUESTIONS" --output "$OUT" "$@"
