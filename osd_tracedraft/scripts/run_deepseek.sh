#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${TARGET:?Set TARGET to the local target model directory}"
: "${DRAFT:?Set DRAFT to its matching EAGLE-3 draft directory}"
: "${OUT:?Set OUT to a new results directory}"
QUESTIONS="${QUESTIONS:-eagle/data/mt_bench/question.jsonl}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
methods=(baseline tracedraft osd osd_tracedraft)
for method in "${methods[@]}"; do
  "$PYTHON" -u -m osd_tracedraft.run \
    --model-profile deepseek --method "$method" \
    --target "$TARGET" --draft "$DRAFT" --questions "$QUESTIONS" \
    --output "$OUT/$method" "$@"
done
"$PYTHON" -m osd_tracedraft.report \
  --runs "$OUT/baseline" "$OUT/tracedraft" "$OUT/osd" "$OUT/osd_tracedraft" \
  --output "$OUT/comparison"
