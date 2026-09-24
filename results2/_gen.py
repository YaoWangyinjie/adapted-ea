# -*- coding: utf-8 -*-
"""Deterministically generate the requested results2 reports from raw P0 data."""
import json
import unicodedata
from pathlib import Path

ROOT = Path("/root/paddlejob/workspace/env_run/yw/adapted-ea/P0-results")
OUT = Path("/root/paddlejob/workspace/env_run/yw/adapted-ea/results2")
SEL = ROOT / "selected_e34_20260923_014102"
SHORT = ROOT / "targeted_short_20260924_023906/targeted_short/runs"
FOLLOW = ROOT / "tracedraft_followup_20260924_132428"
GPU = "NVIDIA A800"
GENERATED = []
PARSED = set()


def width(value):
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in str(value))


def table(headers, rows):
    widths = [width(item) for item in headers]
    for row in rows:
        for index, item in enumerate(row):
            widths[index] = max(widths[index], width(item))
    def render(row):
        return "  ".join(
            (str(item).ljust(widths[index]) if index == 0 else str(item).rjust(widths[index]))
            for index, item in enumerate(row)
        )
    return "\n".join([render(headers), "  ".join("-" * n for n in widths)] + [render(row) for row in rows])


def read_json(path):
    path = Path(path)
    PARSED.add(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    path = Path(path)
    PARSED.add(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def complete_rows(path):
    rows = read_jsonl(path)
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"source is not complete: {path}")
    required = {"total_accept_length", "total_drafted_tokens", "total_steps", "new_tokens", "generation_seconds"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"missing fields {sorted(missing)}: {path}")
    return rows


def key(row):
    for name in ("stream_index", "qid", "question_id"):
        if name in row:
            return str(row[name])
    raise ValueError("row has no alignment key")


def metric(row):
    elapsed = row["generation_seconds"] + row.get("reset_seconds", 0)
    return (100 * row["total_accept_length"] / row["total_drafted_tokens"],
            row["total_accept_length"] / row["total_steps"], row["new_tokens"] / elapsed)


def pooled(rows):
    accepted = sum(row["total_accept_length"] for row in rows)
    drafted = sum(row["total_drafted_tokens"] for row in rows)
    steps = sum(row["total_steps"] for row in rows)
    tokens = sum(row["new_tokens"] for row in rows)
    seconds = sum(row["generation_seconds"] + row.get("reset_seconds", 0) for row in rows)
    return {"acc": 100 * accepted / drafted, "avg": accepted / steps,
            "tps": tokens / seconds, "tokens": tokens, "seconds": seconds}


def intersect(groups):
    shared = set.intersection(*(set(map(key, rows)) for rows in groups.values()))
    if not shared:
        raise ValueError("empty source intersection")
    return {name: [row for row in rows if key(row) in shared] for name, rows in groups.items()}


def gain(value, base):
    return 100 * (value - base) / base


def sections(groups, order, base="baseline", extra=None):
    groups = intersect(groups)
    aggregates = {name: pooled(groups[name]) for name in order}
    absolute = [[name, f"{data['acc']:.3f}", f"{data['avg']:.4f}", f"{data['tps']:.2f}",
                 str(data["tokens"]), f"{data['seconds']:.1f}"] for name, data in aggregates.items()]
    relative = [[f"{name} vs {base}", f"{gain(aggregates[name]['acc'], aggregates[base]['acc']):+.2f}",
                 f"{gain(aggregates[name]['avg'], aggregates[base]['avg']):+.2f}",
                 f"{gain(aggregates[name]['tps'], aggregates[base]['tps']):+.2f}"]
                for name in order if name != base]
    per_question = {name: {key(row): metric(row) for row in groups[name]} for name in order}
    maximum = []
    for label, index in (("接受率", 0), ("平均接受长度", 1), ("吞吐速度", 2)):
        candidates = []
        for name in order:
            if name == base:
                continue
            for qid, values in per_question[name].items():
                base_value = per_question[base][qid][index]
                if base_value > 0:
                    candidates.append((gain(values[index], base_value), name, qid, base_value, values[index]))
        best = max(candidates)
        maximum.append([label, best[1], best[2], f"{best[3]:.4f}", f"{best[4]:.4f}", f"{best[0]:+.2f}"])
    result = ["一、平均指标\n" + table(["方法", "接受率%", "平均接受长度", "吞吐速度(tok/s)", "吞吐总量(tok)", "时间(s)"], absolute),
              "二、相对基线提升\n" + table(["对比", "接受率提升%", "平均长度提升%", "吞吐提升%"], relative),
              "三、最大单题提升\n" + table(["指标", "方法", "题号", "基线", "实验值", "相对提升%"], maximum)]
    if extra:
        result.append(extra(groups))
    return result


PARAM_KEYS = ("max_new_tokens", "max_length", "dtype", "total_token", "depth", "top_k", "seed",
              "learning_rate", "source", "prompt_weight", "max_prompt_positions", "max_response_positions",
              "warmup_steps", "update_every", "train_steps", "scope", "precision", "validation_fraction",
              "gradient_clip", "max_step_drift", "anchor_weight", "replay_mib", "replay_positions",
              "max_successful_updates")


def parameters(manifests, labels):
    rows = []
    for label, path in zip(labels, manifests):
        args = read_json(path).get("args", {})
        rows.append([label] + [str(args.get(item, "-")) for item in PARAM_KEYS])
    return "实验参数\n" + table(["方法"] + list(PARAM_KEYS), rows)


def report(relative, dataset, groups, order, manifests, labels=None, extra=None, base="baseline"):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    body = sections(groups, order, base=base, extra=extra)
    body.append(parameters(manifests, labels or order))
    target.write_text(f"dataset: {dataset}\n显卡: {GPU}\n\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    GENERATED.append(target)


def paths(base, mapping):
    return {label: complete_rows(Path(base) / sub / "stats.jsonl") for label, sub in mapping.items()}


def manifests(base, mapping):
    return [Path(base) / sub / "manifest.json" for sub in mapping.values()]


def gen_basic():
    aime = SEL / "side/runs/aime_2024"
    amap = {"baseline": "baseline", "Reset": "reset", "P": "persistent.attempt_02"}
    report("1-basic/aime-2024-basic.txt", "AIME-2024", paths(aime, amap), list(amap), manifests(aime, amap))
    live = SEL / "side/livecodebench_16384/runs"
    lmap = {"baseline": "baseline.attempt_01", "Reset": "reset", "P": "persistent"}
    report("1-basic/livecodebench-basic.txt", "LiveCodeBench", paths(live, lmap), list(lmap), manifests(live, lmap))


def gen_pa():
    aime = SEL / "side/runs/aime_2024"
    groups = paths(aime, {"baseline": "baseline", "Reset": "reset", "P": "persistent.attempt_02"})
    groups["P+A"] = complete_rows(SHORT / "aime/p_a/attempt_01/stats.jsonl")
    mfs = [aime / "baseline/manifest.json", aime / "reset/manifest.json",
           aime / "persistent.attempt_02/manifest.json", SHORT / "aime/p_a/attempt_01/manifest.json"]
    report("3-p-a/aime-2024-p-a.txt", "AIME-2024", groups, ["baseline", "Reset", "P", "P+A"], mfs)
    streams = SEL / "runs/streams/pubmedqa500"
    for order in (0, 1):
        base = streams / f"order_{order}"
        mapping = {"baseline": "baseline", "Reset": "reset_f1", "P": "p_f1", "P+A": "p_a"}
        if order == 0:
            mapping["P"] = "p_f1.attempt_03"
            mapping["P+A"] = "p_a.attempt_03"
        report(f"3-p-a/pubmedqa500-order{order}-p-a.txt", f"PubMedQA500 order {order}",
               paths(base, mapping), list(mapping), manifests(base, mapping))


def gen_freeze():
    p_once = SEL / "runs/p_once"
    for dataset, slug in (("MT-Bench", "mt-bench"), ("PubMedQA500", "pubmedqa500")):
        key_name = "mt_bench" if slug == "mt-bench" else "pubmedqa"
        groups = {}
        mfs = []
        order = []
        for seed in (0, 1, 2):
            name = f"P-once seed{seed}"
            run = p_once / key_name / f"seed_{seed}"
            groups[name] = complete_rows(run / "stats.jsonl")
            mfs.append(run / "manifest.json")
            order.append(name)
        # Seed 0 is the comparison baseline when the experiment consists solely of three P-once seeds.
        report(f"4-freeze/{slug}-freeze.txt", dataset, groups, order, mfs, base=order[0])
    aime = SEL / "side/runs/aime_2024"
    groups = paths(aime, {"baseline": "baseline", "P": "persistent.attempt_02"})
    groups["P-once"] = complete_rows(SHORT / "aime/p_once/attempt_01/stats.jsonl")
    mfs = [aime / "baseline/manifest.json", aime / "persistent.attempt_02/manifest.json",
           SHORT / "aime/p_once/attempt_01/manifest.json"]
    report("4-freeze/aime-2024-freeze.txt", "AIME-2024", groups,
           ["baseline", "P", "P-once"], mfs)


def switch_segments(groups):
    rows = []
    for name, values in groups.items():
        for label, lo, hi in (("PubMedQA100", 0, 99), ("TheoremQA100", 100, 199)):
            data = pooled([row for row in values if lo <= int(row["stream_index"]) <= hi])
            rows.append([name, label, f"{data['acc']:.3f}", f"{data['avg']:.4f}", f"{data['tps']:.2f}"])
    return "四、分段指标\n" + table(["方法", "数据段", "接受率%", "平均接受长度", "吞吐速度(tok/s)"], rows)


def gen_switch():
    root = SEL / "runs/streams/pubmedqa100_theoremqa100"
    for order_id in (0, 1, 2):
        base = root / f"order_{order_id}"
        mapping = {"baseline": "baseline", "Reset": "reset_f1", "P": "p_f1", "P+A": "p_a"}
        groups = paths(base, mapping)
        target = OUT / f"5-switch/pubmedqa100-theoremqa100-order{order_id}-switch.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        body = sections(groups, list(mapping), extra=switch_segments)
        body.append(parameters(manifests(base, mapping), list(mapping)))
        target.write_text(f"dataset: PubMedQA100 -> TheoremQA100 order {order_id}\n显卡: {GPU}\n\n" +
                          "\n\n".join(body) + "\n", encoding="utf-8")
        GENERATED.append(target)


def gen_frequency():
    aime = SEL / "side/runs/aime_2024"
    groups = {
        "baseline": complete_rows(aime / "baseline/stats.jsonl")[:20],
        "every1": complete_rows(aime / "persistent.attempt_02/stats.jsonl")[:20],
        "every8": complete_rows(FOLLOW / "runs/aime/frequency/persistent_every8/attempt_01/stats.jsonl"),
        "every16": complete_rows(FOLLOW / "runs/aime/frequency/persistent_every16/attempt_01/stats.jsonl"),
    }
    mfs = [aime / "baseline/manifest.json", aime / "persistent.attempt_02/manifest.json",
           FOLLOW / "runs/aime/frequency/persistent_every8/attempt_01/manifest.json",
           FOLLOW / "runs/aime/frequency/persistent_every16/attempt_01/manifest.json"]
    report("6-frequecy/aime-2024-frequency.txt", "AIME-2024", groups,
           ["baseline", "every1", "every8", "every16"], mfs)


def gen_prefix():
    root = FOLLOW / "runs/aime/prefix_source"
    mapping = {"current": "current_verified_prefix", "causal prior": "prior_verified_prefix_shuffled"}
    base = root / mapping["current"] / "attempt_01"
    prior = root / mapping["causal prior"] / "attempt_01"
    report("7-prefix/aime-2024-prefix.txt", "AIME-2024",
           {"current": complete_rows(base / "stats.jsonl"),
            "causal prior": complete_rows(prior / "stats.jsonl")},
           ["current", "causal prior"], [base / "manifest.json", prior / "manifest.json"],
           base="current")


def validate():
    forbidden = ("p_r", "p+r", "p＋r", "p_r_a", "exactness", "精确性", "pending", "failed",
                 "unfinished", "未完成", "失败", "osd", "onlinespec")
    for path in GENERATED:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) < 2 or not lines[0].startswith("dataset: ") or lines[1] != f"显卡: {GPU}":
            raise AssertionError(f"bad header: {path}")
        lowered = text.lower()
        hits = [item for item in forbidden if item in lowered]
        if hits:
            raise AssertionError(f"forbidden text {hits}: {path}")
    return len(GENERATED), len(PARSED)


def main():
    gen_basic()
    gen_pa()
    gen_freeze()
    gen_switch()
    gen_frequency()
    gen_prefix()
    reports, sources = validate()
    print(f"generated={reports} parsed_sources={sources}")
    for path in GENERATED:
        print(path)


if __name__ == "__main__":
    main()
