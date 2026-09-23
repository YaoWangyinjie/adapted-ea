#!/usr/bin/env bash
# MT-Bench: all 80 questions / 160 turns, EA4 persistent, learning rate 1e-6.
# Run: bash run_mtbench_ea4_persistent.sh
# Override paths/GPU with EA4_BASE, EA4_DRAFT, EA4_OUTPUT_DIR, EA4_PYTHON,
# and CUDA_VISIBLE_DEVICES. Use --dry-run to print the command without running.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EA4_PYTHON="${EA4_PYTHON:-python}"
# adapted-ea/ and new-eagle/ are siblings; resolve from this script's location.
MODEL_WEIGHT_DIR="$(dirname -- "$SCRIPT_DIR")/new-eagle/model_weight"
EA4_BASE="${EA4_BASE:-$MODEL_WEIGHT_DIR/DeepSeek-R1-Distill-Llama-8B}"
EA4_DRAFT="${EA4_DRAFT:-$MODEL_WEIGHT_DIR/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B}"
EA4_OUTPUT_DIR="${EA4_OUTPUT_DIR:-$SCRIPT_DIR/results/mtbench_ea4_persistent_lr1e-6_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
QUESTION_FILE="$SCRIPT_DIR/eagle/data/mt_bench/question.jsonl"

command=(
  "$EA4_PYTHON" -u -m eagle.evaluation.gen_ea_answer_ea4
  --base-model-path "$EA4_BASE"
  --ea-model-path "$EA4_DRAFT"
  --question-file "$QUESTION_FILE"
  --output-dir "$EA4_OUTPUT_DIR"
  --experiment-name mtbench_ea4_persistent_lr1e-6
  --weight-mode persistent
  --learning-rate 1e-6
  --question-begin 0
  --question-end 80
  --max-new-tokens 1024
  --max-length 4096
  --total-token 60
  --depth 5
  --top-k 10
  --dtype float16
  --seed 0
  --warmup-runs 2
  --verify-greedy 0
  --scope head_only
  --precision master_fp32
  --alignment correct
  --update-at warmup
  --warmup-steps 5
  --update-every 1
  --source balanced
  --prompt-weight 0.25
  --max-prompt-positions 64
  --max-response-positions 64
  --train-steps 1
  --chunk-size 32
  --feature-chunk-size 256
  --distill-temperature 1.0
  --validation-fraction 0.2
  --gradient-clip 0.2
  --max-step-drift 0.02
  --max-anchor-drift 0
  --anchor-weight 0
  --replay-mib 0
)

if [[ "$#" -eq 1 && "$1" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
elif [[ "$#" -ne 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

if [[ -e "$EA4_OUTPUT_DIR" || -e "${EA4_OUTPUT_DIR}.log" ]]; then
  printf 'Refusing to overwrite existing output/log: %s\n' "$EA4_OUTPUT_DIR" >&2
  exit 1
fi

"$EA4_PYTHON" - "$QUESTION_FILE" "$EA4_BASE" "$EA4_DRAFT" <<'PY'
import json
import sys
from pathlib import Path
import torch

for directory in sys.argv[2:]:
    assert (Path(directory) / "config.json").is_file(), f"Missing model config: {directory}"
questions = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
assert len(questions) == 80, f"Expected 80 questions, got {len(questions)}"
assert len({str(q["question_id"]) for q in questions}) == 80, "Duplicate question IDs"
assert all(isinstance(q["turns"], list) and len(q["turns"]) == 2 for q in questions), "Expected two turns per question"
assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"
print("Preflight OK: 80 questions, 160 turns; GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p -- "$(dirname -- "$EA4_OUTPUT_DIR")"
printf 'Results: %s\nLog: %s.log\n' "$EA4_OUTPUT_DIR" "$EA4_OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "${EA4_OUTPUT_DIR}.log"

"$EA4_PYTHON" - "$EA4_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
def read_jsonl(name):
    return [json.loads(line) for line in (root / name).read_text(encoding="utf8").splitlines() if line.strip()]
assert (root / "summary.json").is_file(), "Run did not finish normally"
answers, stats = read_jsonl("answers.jsonl"), read_jsonl("stats.jsonl")
assert len(answers) == 80 and len({str(a["question_id"]) for a in answers}) == 80, "Incomplete question coverage"
assert all(a["complete"] and len(a["choices"][0]["turns"]) == 2 for a in answers), "Incomplete answers"
assert len(stats) == 160 and all(s["status"] == "ok" for s in stats), "Failed/skipped turns"
assert len({(str(s["qid"]), s["turn"]) for s in stats}) == 160, "Duplicate turns"
summary = json.loads((root / "summary.json").read_text(encoding="utf8"))
print("Completed: all 80 questions / 160 turns.")
print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
PY
