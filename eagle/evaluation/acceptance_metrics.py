"""Read-only acceptance accounting. Standard library only; no models or GPUs.

Run from adapted-ea:
  python -m eagle.evaluation.acceptance_metrics --stats results/mtbench_noupdate_stats.jsonl --output audit.json
  python -m eagle.evaluation.acceptance_metrics --sft-results path/to/results.jsonl --idx-semantics zero-based

Lengths are deliberately named: draft children, verified extension including one
root per step, and actually returned tokens after truncation are different.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def count_speculative_steps(idx, stats=None, idx_semantics="zero-based"):
    """Prefer an explicit count; otherwise interpret the documented legacy idx."""
    if stats is not None and "total_steps" in stats:
        return _count(stats["total_steps"], "total_steps")
    if isinstance(idx, bool) or not isinstance(idx, int):
        raise ValueError("idx must be an integer")
    if idx_semantics == "zero-based":
        return _count(idx + 1, "idx + 1")
    if idx_semantics == "count":
        return _count(idx, "idx")
    raise ValueError("idx_semantics must be zero-based or count")


def _ratio(n, d):
    return n / d if d else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_stats(rows):
    groups = defaultdict(lambda: {"accepted": 0, "steps": 0, "returned": 0, "turns": 0})
    a = s = n = turns = skipped = 0
    seen = set()
    for line, row in enumerate(rows, 1):
        if row.get("status") not in (None, "ok", "success"):
            skipped += 1
            continue
        aa = _count(row["total_accept_length"], "total_accept_length")
        ss = _count(row["total_steps"], "total_steps")
        nn = _count(row["new_tokens"], "new_tokens")
        if (ss == 0 and (aa or nn)) or nn > aa + ss:
            raise ValueError(f"Line {line}: inconsistent token/step counts")
        qid = row.get("question_id", row.get("qid", line))
        key = (row.get("experiment", ""), row.get("model_id", ""),
               str(qid), row.get("choice", 0))
        turnkey = key + (row.get("turn", line),)
        if turnkey in seen:
            raise ValueError(f"Line {line}: duplicate question/choice/turn; do not mix runs")
        seen.add(turnkey)
        g = groups[key]
        g["accepted"] += aa
        g["steps"] += ss
        g["returned"] += nn
        g["turns"] += 1
        a += aa
        s += ss
        n += nn
        turns += 1
    gs = list(groups.values())
    return {
        "turns": turns, "questions_or_choices": len(gs), "skipped_rows": skipped,
        "accepted_children": a, "verification_steps": s, "returned_tokens": n,
        "discarded_verified_tokens": a + s - n,
        "micro": {
            "accepted_children_per_step": _ratio(a, s),
            "verified_tokens_per_step_including_root": _ratio(a + s, s),
            "returned_tokens_per_step": _ratio(n, s),
            "discarded_tokens_per_step": _ratio(a + s - n, s),
        },
        "macro_question": {
            "accepted_children_per_step": _mean(_ratio(g["accepted"], g["steps"]) for g in gs),
            "verified_tokens_per_step_including_root": _mean(_ratio(g["accepted"] + g["steps"], g["steps"]) for g in gs),
            "returned_tokens_per_step": _mean(_ratio(g["returned"], g["steps"]) for g in gs),
        },
        "legacy_zero_based_denominator_demo": {
            "pretruncation_numerator_macro": _mean(
                _ratio(g["accepted"] + g["steps"], g["steps"] - g["turns"]) for g in gs
                if g["steps"] > g["turns"]),
            "returned_numerator_macro": _mean(
                _ratio(g["returned"], g["steps"] - g["turns"]) for g in gs
                if g["steps"] > g["turns"]),
            "undefined_questions": sum(g["steps"] <= g["turns"] for g in gs),
            "note": "Demonstrates the old sum(idx) convention; these are not corrected lengths.",
        },
    }


def summarize_sft(rows, idx_semantics):
    """Recount existing per-question counters, preserving the original numerator."""
    groups = defaultdict(list)
    for row in rows:
        n = _count(row["new_tokens"], "new_tokens")
        if "speculative_steps" in row:
            s = _count(row["speculative_steps"], "speculative_steps")
        else:
            if "idxs" not in row:
                raise ValueError("SFT recount requires idxs or speculative_steps")
            s = sum(count_speculative_steps(i, idx_semantics=idx_semantics) for i in row["idxs"])
        if n and not s:
            raise ValueError("Nonzero new_tokens with zero steps")
        groups[row.get("model_tag", "unspecified")].append((n, s, row.get("avg_accept_len")))
    return {tag: {
        "questions": len(xs),
        "model_counter_tokens": sum(x[0] for x in xs),
        "verification_steps": sum(x[1] for x in xs),
        "counter_tokens_per_step_micro": _ratio(sum(x[0] for x in xs), sum(x[1] for x in xs)),
        "counter_tokens_per_step_macro": _mean(_ratio(x[0], x[1]) for x in xs),
        "archived_avg_accept_macro": _mean(x[2] for x in xs),
        "numerator_scope": "Original model new_token counter; cannot infer EOS/length trimming from this field alone.",
        "legacy_idx_semantics": idx_semantics,
    } for tag, xs in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stats", type=Path)
    g.add_argument("--sft-results", type=Path)
    p.add_argument("--idx-semantics", choices=["zero-based", "count"], default="zero-based")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    src = args.stats or args.sft_results
    if args.output and src.resolve() == args.output.resolve():
        p.error("Output must not overwrite the input log")
    rows = [json.loads(line) for line in src.read_text("utf-8-sig").splitlines() if line.strip()]
    report = {
        "input": str(src), "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "summary": summarize_stats(rows) if args.stats else summarize_sft(rows, args.idx_semantics),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf8")
    print(rendered)


if __name__ == "__main__":
    main()
