"""Paired raw-count aggregation; charge every learner's training time."""
import argparse
import csv
import time
from pathlib import Path
from .core.io_utils import read_json,read_jsonl,write_json,write_jsonl,token_hash,sha256
from .core.metrics import summarize,gains

def aggregate(directory):
    root=Path(directory);state=read_json(root/"state.json");config=read_json(root/"config.json")
    rows=[];train_seconds=merge_seconds=0.;steps=0;block_summaries=[];history=[]
    outputs={name:[] for name in ("answers","training_records","trace_updates")}
    for b in state["blocks"]:
        batch=read_jsonl(root/b["infer_directory"]/"turns.jsonl");rows.extend(batch)
        for name in outputs:
            outputs[name].extend(read_jsonl(root/b["infer_directory"]/(name+".jsonl")))
        training=[read_json(root/p/"report.json") for p in b["learner_directories"]]
        train=sum(x["stage_seconds"] for x in training)
        merged=read_json(root/b["merge_directory"]/"report.json")
        merge=merged["stage_seconds"]
        train_seconds+=train;merge_seconds+=merge;steps+=sum(x["steps"] for x in training)
        bs=summarize(batch,train+merge);bs.pop("osd_training_seconds")
        block_summaries.append(dict(block=b["index"],outer_training_seconds=train,ensemble_merge_seconds=merge,**bs))
        history.append(dict(block=b["index"],weights=b["weights"],meta_losses=b["meta_losses"],
            cumulative_losses=b["cumulative_next"],training_parents=b["training_parents"],
            merged_sha256=merged["weights_sha256"]))
    summary=summarize(rows,train_seconds+merge_seconds,state["pipeline_seconds"])
    summary.pop("osd_training_seconds")
    summary.update(status=state["status"],method=config["method"],variant=config["variant"],
        request_count=len({r["stream_index"] for r in rows}),outer_training_seconds=train_seconds,
        ensemble_merge_seconds=merge_seconds,outer_optimizer_steps=steps,
        prompt_truncated_turns=sum(r["prompt_tokens_removed"]>0 for r in rows),
        generated_tokens_sha256=token_hash([(r["question_id"],r["turn"],r["generated_sha256"]) for r in rows]),
        prompts_sha256=token_hash([(r["question_id"],r["turn"],r["prompt_sha256"]) for r in rows]))
    windows=[]
    for start in range(0,1+max((r["stream_index"] for r in rows),default=-1),100):
        subset=[r for r in rows if start<=r["stream_index"]<start+100]
        window=summarize(subset);window.pop("osd_training_seconds")
        windows.append(dict(request_start=start+1,request_end=max(r["stream_index"] for r in subset)+1,**window))
    write_json(root/"summary.json",summary);write_json(root/"block_summaries.json",block_summaries)
    write_json(root/"ensemble_history.json",history);write_json(root/"windows_100.json",windows)
    for name,records in outputs.items():
        write_jsonl(root/(name+".jsonl"),records)
    return summary

def write_output_failure(reference,candidate,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    a=read_jsonl(reference/"answers.jsonl");b=read_jsonl(candidate/"answers.jsonl")
    right={(str(r["question_id"]),r["turn"]):r for r in b}
    changes=[]
    for row in a:
        key=(str(row["question_id"]),row["turn"])
        other=right.get(key)
        if other is None:
            changes.append(dict(question_id=row["question_id"],turn=row["turn"],reason="missing_turn"))
            continue
        x=row["generated_ids"];y=other["generated_ids"]
        diffs=[i for i in range(max(len(x),len(y))) if (x[i] if i<len(x) else None)!=(y[i] if i<len(y) else None)]
        if diffs:
            changes.append(dict(question_id=row["question_id"],turn=row["turn"],
                reference_tokens=len(x),candidate_tokens=len(y),different_positions=len(diffs),
                first_differences=[dict(index=i,reference=x[i] if i<len(x) else None,
                    candidate=y[i] if i<len(y) else None) for i in diffs[:16]]))
    # Avoid leaving an old successful report next to a newly failed comparison.
    for name in ("comparison.md","comparison.csv","comparison.json"):
        old=out/name
        if old.exists():old.rename(old.with_name(old.name+".previous-"+str(time.time_ns())))
    write_json(out/"comparison_failure.json",dict(status="invalid_pair",reason="prompt_or_output_mismatch",
        speedup_reported=False,reference=str(reference.resolve()),candidate=str(candidate.resolve()),
        mismatched_turns=changes,
        explanation="Full outputs differ. Inspect raw answers and target precision; no speedup is emitted."))


def compare(run_dirs,output,reference="onlinespec"):
    if reference not in ("onlinespec","eagle3","tracedraft"):
        raise ValueError("Reference must be onlinespec, eagle3 or tracedraft")
    runs=[]
    for path in run_dirs:
        root=Path(path);summary=aggregate(root)
        if summary["status"]!="complete":
            raise ValueError(f"Incomplete run: {root}")
        runs.append((root,read_json(root/"config.json"),read_json(root/"manifest.json"),summary))
    refs=[r for r in runs if r[1]["method"]==reference]
    if len(refs)!=1 or len({r[1]["method"] for r in runs})!=len(runs):
        raise ValueError("Supply one run per method and exactly one reference")
    base=refs[0];results=[]
    for root,config,manifest,summary in runs:
        ignore={"method","output"}
        mismatch=[k for k in set(config)|set(base[1]) if k not in ignore and config.get(k)!=base[1].get(k)]
        for key in ("target_files","draft_files","source_files","questions_sha256"):
            if manifest[key]!=base[2][key]:
                mismatch.append(key)
        if mismatch:
            raise ValueError(f"Unmatched runs: {mismatch}")
        if summary["generated_tokens_sha256"]!=base[3]["generated_tokens_sha256"] or summary["prompts_sha256"]!=base[3]["prompts_sha256"]:
            write_output_failure(base[0],root,output)
            raise ValueError("Prompt/output token sequences differ; see comparison_failure.json; no speedup is reported")
        results.append(dict(run_directory=str(root.resolve()),**summary,**gains(base[3],summary)))
    outer={}
    for root,config,_,_ in runs:
        if config["method"] not in ("onlinespec","onlinespec_trace"):
            continue
        fingerprint=[]
        def verified_weight(path,expected):
            actual=sha256(Path(path)/"model.safetensors")
            if actual!=expected:
                raise ValueError("Outer trajectories cannot be verified: checkpoint bytes changed")
            return actual
        for b in read_json(root/"state.json")["blocks"]:
            fingerprint.append(dict(block=b["index"],weights=b["weights"],losses=b["meta_losses"],
                merged=verified_weight(root/b["merge_directory"]/"checkpoint",read_json(root/b["merge_directory"]/"report.json")["weights_sha256"]),
                learners=[verified_weight(root/p/"checkpoint",read_json(root/p/"checkpoint/provenance.json")["weights_sha256"]) for p in b["learner_directories"]]))
        outer[config["method"]]=fingerprint
    matched_outer=None
    if len(outer)==2:
        matched_outer=outer["onlinespec"]==outer["onlinespec_trace"]
        if not matched_outer:
            raise ValueError("Outer trajectories differ; this is not an isolated TraceDraft comparison")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    old_failure=out/"comparison_failure.json"
    if old_failure.exists():old_failure.rename(old_failure.with_name(old_failure.name+".previous-"+str(time.time_ns())))
    by_method={r["method"]:r for r in results}
    pairwise={}
    if "onlinespec_trace" in by_method:
        for name in ("onlinespec","tracedraft"):
            if name in by_method:
                pairwise["onlinespec_trace_vs_"+name]=gains(by_method[name],by_method["onlinespec_trace"])
    write_json(out/"comparison.json",dict(reference_method=reference,runs=results,
        pairwise_gains=pairwise,outer_trajectory_identical=matched_outer))
    keys=["method","variant","acceptance_percent","acceptance_delta_pp","average_length",
          "average_length_relative_gain_percent","average_length_including_root","returned_tokens_per_step",
          "new_tokens","tokens_per_second_generation","speedup_generation",
          "tokens_per_second_training_inclusive","speedup_training_inclusive",
          "tokens_per_second_pipeline","speedup_pipeline","outer_training_seconds","ensemble_merge_seconds",
          "outer_optimizer_steps","trace_updated_turns","trace_rolled_back_turns","trace_validation_rejected_turns"]
    with (out/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");writer.writeheader();writer.writerows(results)
    lines=["# OnlineSPEC / TraceDraft comparison","",f"Target model: {base[1]['target']}",
        f"OnlineSPEC variant: {base[1]['variant']}; merge precision: {base[1]['merge_precision']}.","",
        "| Method | Acceptance (%) | Δacc (pp) | Length (children) | Length gain (%) | Generation tok/s | Ratio | Including training tok/s | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        columns=["acceptance_percent","acceptance_delta_pp","average_length","average_length_relative_gain_percent",
                 "tokens_per_second_generation","speedup_generation","tokens_per_second_training_inclusive","speedup_training_inclusive"]
        lines.append("| "+r["method"]+" | "+" | ".join(f"{r[k]:.4f}" for k in columns)+" |")
    for name, delta in pairwise.items():
        lines += ["", f"{name}: generation ratio {delta['speedup_generation']:.4f}; "
                  f"training-inclusive ratio {delta['speedup_training_inclusive']:.4f}."]
    lines+=["",f"Ratios use the matched {reference} run, not autoregressive decoding.",
        "Generation time includes TraceDraft collection, features and online updates. The training-inclusive denominator additionally charges resets, every learner's training/checkpoint stage, and ensemble merging.",
        "Pipeline time also includes all model reloads, warmup, correctness probes and process/file overhead.",
        "Length excludes the root; +root and returned-token length are available in CSV/JSON. Acceptance uses the local EA4 candidate-path-width denominator.",
        f"Outer learner checkpoints, weights and meta losses identical: {matched_outer}.",
        "All compared runs have identical prompts and full generated token sequences."]
    (out/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--output");p.add_argument("--reference",default="onlinespec",choices=["onlinespec","eagle3","tracedraft"])
    args=p.parse_args()
    if args.output:
        compare(args.runs,args.output,args.reference)
    else:
        for root in args.runs:
            print(aggregate(root))
if __name__=="__main__":
    main()
