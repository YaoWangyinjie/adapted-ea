import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from onlinespec_trace import run as runner
from onlinespec_trace import worker
from onlinespec_trace.config import Settings
from onlinespec_trace.report import compare,aggregate
from onlinespec_trace.core.checkpoints import save_checkpoint,read_checkpoint
from onlinespec_trace.core.models import build_draft
from onlinespec_trace.core.backend.ea_model_4 import EaModel
from onlinespec_trace.core.io_utils import read_json,read_jsonl,write_json,write_jsonl

def environment(tiny,tmp_path,monkeypatch):
    original,raw=tiny
    target=tmp_path/"target";target.mkdir();write_json(target/"config.json",{"test":True})
    draft=tmp_path/"draft";save_checkpoint(draft,original.ea_layer,raw)
    data=tmp_path/"questions.jsonl"
    write_jsonl(data,[dict(question_id=i,turns=["Hi","Why"] if i==0 else ["Different"]) for i in range(5)])
    s=Settings(model_profile="deepseek",target=str(target),draft=str(draft),questions=str(data),
        output=str(tmp_path/"run"),max_length=96,max_new_tokens=12,total_token=6,depth=2,top_k=2,
        block_size=2,epochs=1,batch_size=2,dtype="float32",draft_device="cpu",cpu_threads=1,
        warmup_runs=0,verify_greedy=5,verify_tokens=12,trace_update_round=1,
        trace_validation_guard=False,lr_1=2e-4,lr_2=1e-4,lr_3=4e-4).validate()
    def load(inner,path,training=False):
        cfg,state=read_checkpoint(path)
        base=copy.deepcopy(original.base_model)
        d=build_draft(cfg,state,base.model.embed_tokens.weight,inner,torch.device("cpu"),training)
        return EaModel(base,d,original.tokenizer),cfg
    monkeypatch.setattr(worker,"load_model",load)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _: "tiny-test CPU")
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _: SimpleNamespace(total_memory=0))
    def local_stage(job,root):
        runner.fresh_directory(job["directory"],root)
        write_json(Path(job["directory"])/"job.json",job)
        return worker.execute(job)
    monkeypatch.setattr(runner,"run_stage",local_stage)
    return s,local_stage

@pytest.mark.parametrize("variant",["hedge","ens"])
def test_full_pair_has_identical_outer_models_and_correct_accounting(tiny,tmp_path,monkeypatch,variant):
    s,_=environment(tiny,tmp_path,monkeypatch)
    paths=[]
    for method in ("onlinespec","onlinespec_trace"):
        cfg=replace(s,variant=variant,method=method,output=str(tmp_path/method))
        summary=runner.run(cfg);paths.append(cfg.output)
        assert summary["status"]=="complete" and summary["request_count"]==5
        assert summary["outer_optimizer_steps"]>0
        assert all(r["greedy_probe_passed"] for b in read_json(Path(cfg.output)/"state.json")["blocks"]
                   for r in read_jsonl(Path(cfg.output)/b["infer_directory"]/"turns.jsonl"))
        state=read_json(Path(cfg.output)/"state.json")
        assert len(state["blocks"])==3
        assert state["blocks"][-1]["learner_directories"]==[]
        second=state["blocks"][1]
        if variant=="hedge":
            assert second["training_parents"]==[second["merged_checkpoint"]]*3
        else:
            assert second["training_parents"]==state["blocks"][0]["bases_next"]
        expected=sum(read_json(Path(cfg.output)/p/"report.json")["stage_seconds"]
                     for b in state["blocks"] for p in b["learner_directories"])
        assert summary["outer_training_seconds"]==expected
        assert summary["training_inclusive_seconds"]==pytest.approx(summary["generation_seconds"]+
            summary["reset_seconds"]+expected+summary["ensemble_merge_seconds"])
        assert runner.run(cfg,resume=True)==summary
    compare(paths,tmp_path/"comparison")
    assert read_json(tmp_path/"comparison/comparison.json")["outer_trajectory_identical"]
    rows=read_jsonl(Path(paths[1])/"trace_updates.jsonl")
    assert any(r["adaptation"]["status"]=="updated" for r in rows)
    # Every production record is labeled with the new method, not the core alias.
    assert all(r["method"]=="onlinespec_trace" for r in read_jsonl(Path(paths[1])/"answers.jsonl"))
    if variant=="ens":
        opt=torch.load(Path(paths[0])/"blocks/0001/learner_0/checkpoint/optimizer.pt",weights_only=True)
        assert all(v["step"]>=2 for v in opt["state"].values())

def test_resume_restarts_only_unfinished_block(tiny,tmp_path,monkeypatch):
    s,stage=environment(tiny,tmp_path,monkeypatch)
    called=[]
    def interrupted(job,root):
        called.append((job["block_index"],job["stage"],job.get("learner")))
        if job["block_index"]==1 and job.get("learner")==1:
            raise RuntimeError("Injected interruption")
        return stage(job,root)
    monkeypatch.setattr(runner,"run_stage",interrupted)
    with pytest.raises(RuntimeError,match="Injected"):
        runner.run(s)
    assert not (Path(s.output)/"RUNNING.lock").exists()
    assert len(read_json(Path(s.output)/"state.json")["blocks"])==1
    called.clear()
    def resumed(job,root):
        called.append(job["block_index"]);return stage(job,root)
    monkeypatch.setattr(runner,"run_stage",resumed)
    report=runner.run(s,resume=True)
    assert report["status"]=="complete" and 0 not in called
    assert len(read_jsonl(Path(s.output)/"answers.jsonl"))==6
    assert list((Path(s.output)/"blocks/0001").glob("*.failed-*"))

def test_comparison_rejects_config_and_checkpoint_drift(tiny,tmp_path,monkeypatch):
    s,_=environment(tiny,tmp_path,monkeypatch)
    paths=[]
    for method in ("onlinespec","onlinespec_trace"):
        cfg=replace(s,method=method,output=str(tmp_path/method))
        runner.run(cfg);paths.append(cfg.output)
    p=Path(paths[1])/"config.json";config=read_json(p);config["trace_lr"]*=2;write_json(p,config)
    with pytest.raises(ValueError,match="Unmatched"):
        compare(paths,tmp_path/"comparison")
    config["trace_lr"]/=2;write_json(p,config)
    r=Path(paths[1])/"blocks/0001/merge/report.json";changed=read_json(r);changed["weights_sha256"]="changed";write_json(r,changed)
    with pytest.raises(ValueError,match="trajectories"):
        compare(paths,tmp_path/"comparison")

def test_dry_run_counts_all_learners(tiny,tmp_path,monkeypatch):
    s,_=environment(tiny,tmp_path,monkeypatch)
    p=runner.run(s,dry_run=True)
    assert p["blocks"]==3 and p["learner_training_jobs"]==6
    assert not Path(s.output).exists()


def test_output_mismatch_emits_diagnostics_without_speedup(tmp_path):
    from onlinespec_trace.report import write_output_failure
    a=tmp_path/"a";b=tmp_path/"b";out=tmp_path/"comparison"
    write_jsonl(a/"answers.jsonl",[dict(question_id=1,turn=0,generated_ids=[1,2,3])])
    write_jsonl(b/"answers.jsonl",[dict(question_id=1,turn=0,generated_ids=[1,7,3])])
    out.mkdir();(out/"comparison.csv").write_text("stale",encoding="utf8")
    write_output_failure(a,b,out)
    r=read_json(out/"comparison_failure.json")
    assert r["status"]=="invalid_pair" and r["speedup_reported"] is False
    assert r["mismatched_turns"][0]["first_differences"]==[dict(index=1,reference=2,candidate=7)]
    assert not (out/"comparison.csv").exists()
    assert len(list(out.glob("comparison.csv.previous-*")))==1


@pytest.mark.parametrize("variant", ["hedge", "ens"])
def test_three_way_runner_keeps_standalone_trace_free_of_outer_training(tiny, tmp_path, monkeypatch, variant):
    from onlinespec_trace.comparison import run_comparison, METHODS
    s, _ = environment(tiny, tmp_path, monkeypatch)
    s = replace(s, variant=variant, output=str(tmp_path / "three_way"))
    results = run_comparison(s)
    root = Path(s.output)
    assert [r["method"] for r in results] == list(METHODS)
    standalone = replace(s, method="tracedraft")
    assert standalone.has_outer is False
    assert standalone.inner().has_trace is True
    assert standalone.inner().has_osd is False
    assert runner.plan(standalone)["learner_training_jobs"] == 0
    path = root / "tracedraft"
    summary = read_json(path / "summary.json")
    assert summary["outer_optimizer_steps"] == 0
    assert summary["outer_training_seconds"] == 0
    assert summary["trace_updated_turns"] > 0
    assert not list(path.glob("blocks/*/learner_*"))
    state = read_json(path / "state.json")
    assert all(b["training_parents"] is None and b["meta_losses"] is None for b in state["blocks"])
    assert all(b["bases_next"] == [s.draft] * 3 for b in state["blocks"])
    assert all(r["method"] == "tracedraft" for r in read_jsonl(path / "answers.jsonl"))
    report = read_json(root / "comparison/comparison.json")
    assert report["outer_trajectory_identical"] is True
    by_method = {r["method"]: r for r in results}
    assert by_method["onlinespec"]["speedup_generation"] == 1.0
    pair = report["pairwise_gains"]["onlinespec_trace_vs_tracedraft"]
    assert pair["speedup_generation"] == pytest.approx(
        by_method["onlinespec_trace"]["tokens_per_second_generation"] /
        by_method["tracedraft"]["tokens_per_second_generation"])
    # The alternative denominator can be selected without rerunning inference.
    compared = compare([root / method for method in METHODS], root / "vs_trace", reference="tracedraft")
    assert next(r for r in compared if r["method"] == "tracedraft")["speedup_generation"] == 1.0
