"""Run OnlineSPEC blocks with immutable checkpoints and boundary resume."""
import argparse
from dataclasses import fields,MISSING
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import Settings
from .core.io_utils import read_json,write_json,write_jsonl,load_questions,sha256,model_files
from .ensemble import merge_checkpoints,weights_for_block,next_history,training_parents

PROJECT=Path(__file__).resolve().parents[1]

def source_manifest():
    return [dict(name=str(p.relative_to(PROJECT)).replace("\\","/"),sha256=sha256(p))
            for p in sorted((PROJECT/"onlinespec_trace").rglob("*.py"))]

def checkpoint_manifest(path):
    result=model_files(path)
    opt=Path(path)/"optimizer.pt"
    if opt.exists():
        result.append(dict(name=opt.name,bytes=opt.stat().st_size,sha256=sha256(opt)))
    return result

def fresh_directory(directory,root):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(Path(root).resolve()) or directory==Path(root).resolve():
        raise ValueError("Stage must be a child of the run output")
    if directory.exists():
        directory.rename(directory.with_name(directory.name+".failed-"+str(time.time_ns())))
    directory.mkdir(parents=True)
    return directory

def run_stage(job,root):
    directory=fresh_directory(job["directory"],root)
    path=directory/"job.json";write_json(path,job)
    with (directory/"console.log").open("w",encoding="utf8") as log:
        subprocess.run([sys.executable,"-u","-m","onlinespec_trace.worker","--job",str(path)],
            cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read_json(directory/"report.json")

def plan(settings):
    questions=load_questions(settings.questions,settings.order_seed)
    blocks=(len(questions)+settings.block_size-1)//settings.block_size
    training_blocks=(blocks if settings.train_last_block else max(0,blocks-1)) if settings.has_outer else 0
    return dict(questions=len(questions),turns=sum(len(q["turns"]) for q in questions),
                blocks=blocks,learner_training_jobs=training_blocks*3,settings=settings.to_dict())

def run(settings,resume=False,dry_run=False):
    s=settings.validate()
    for key in ("target","draft","questions","output"):
        setattr(s,key,str(Path(getattr(s,key)).resolve()))
    task_plan=plan(s)
    if dry_run:
        print(__import__("json").dumps(task_plan,ensure_ascii=False,indent=2))
        return task_plan
    root=Path(s.output)
    if any(root.is_relative_to(Path(p)) for p in (s.target,s.draft)):
        raise ValueError("Output must not be inside an input model directory")
    root.mkdir(parents=True,exist_ok=True)
    lock=root/"RUNNING.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"{lock}: check the recorded process before removing a stale lock")
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.perf_counter();state=None;previous_seconds=0.
    try:
        manifest=dict(questions_sha256=sha256(s.questions),target_files=model_files(s.target),
            draft_files=checkpoint_manifest(s.draft),source_files=source_manifest(),
            upstream_commit="e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b",
            algorithm="OnlineSPEC "+s.variant+" with optional within-response TraceDraft")
        if (root/"config.json").exists():
            if not resume:
                raise FileExistsError("Use a new output directory or --resume")
            if read_json(root/"config.json")!=s.to_dict() or read_json(root/"manifest.json")!=manifest:
                raise ValueError("Resume settings, data, model or source changed")
            state=read_json(root/"state.json");previous_seconds=state["pipeline_seconds"]
            if state["status"]=="complete":
                from .report import aggregate
                return aggregate(root)
        else:
            if any(p.name!="RUNNING.lock" for p in root.iterdir()):
                raise FileExistsError("Output directory is not empty")
            write_json(root/"config.json",s.to_dict());write_json(root/"manifest.json",manifest)
            write_json(root/"plan.json",task_plan)
            state=dict(status="running",blocks=[],pipeline_seconds=0.)
            write_json(root/"state.json",state)
        state.pop("error",None)
        questions=load_questions(s.questions,s.order_seed)
        write_jsonl(root/"question_order.jsonl",questions)
        chunks=[questions[i:i+s.block_size] for i in range(0,len(questions),s.block_size)]
        bases=[s.draft]*3;latest=None;cumulative=[0.]*3;turn_offset=0
        for index,chunk in enumerate(chunks):
            completed=next((b for b in state["blocks"] if b["index"]==index),None)
            if completed:
                for saved in completed["checkpoint_fingerprints"]:
                    if checkpoint_manifest(saved["path"])!=saved["files"]:
                        raise ValueError("Completed learner checkpoint was changed")
                bases=completed["bases_next"];latest=completed["latest_next"];cumulative=completed["cumulative_next"]
                turn_offset+=sum(len(q["turns"]) for q in chunk)
                continue
            blockdir=root/"blocks"/f"{index:04d}";blockdir.mkdir(parents=True,exist_ok=True)
            qpath=blockdir/"questions.jsonl";write_jsonl(qpath,chunk)
            weights=weights_for_block(s.variant,latest,cumulative,s.hedge_temperature,s.ens_epsilon)
            merge_dir=fresh_directory(blockdir/"merge",root)
            start=time.perf_counter()
            merged=merge_dir/"checkpoint"
            merge=merge_checkpoints(bases,weights,merged,s.merge_precision)
            merge_seconds=time.perf_counter()-start
            write_json(merge_dir/"report.json",dict(merge,stage_seconds=merge_seconds,
                latest_losses_used=latest,cumulative_losses_used=cumulative))
            common=dict(settings=s.to_dict(),block_index=index,turn_offset=turn_offset)
            inf=blockdir/"infer"
            print(f"{s.method} {s.variant}: block {index+1}/{len(chunks)} inference",flush=True)
            run_stage(dict(common,stage="infer",directory=str(inf),questions=str(qpath),
                           checkpoint_in=str(merged)),root)
            train_dirs=[];next_bases=bases;losses=None;parents=None
            if s.has_outer and (index<len(chunks)-1 or s.train_last_block):
                parents=training_parents(s.variant,merged,bases)
                next_bases=[];losses=[]
                for learner,parent in enumerate(parents):
                    directory=blockdir/f"learner_{learner}";checkpoint=directory/"checkpoint"
                    print(f"{s.method}: block {index+1}, learner {learner+1}/3, lr={s.learning_rates[learner]}",flush=True)
                    report=run_stage(dict(common,stage="train",learner=learner,directory=str(directory),
                        records=str(inf/"training_records.jsonl"),checkpoint_in=parent,checkpoint_out=str(checkpoint)),root)
                    losses.append(report["meta_loss"]);next_bases.append(str(checkpoint))
                    train_dirs.append(str(directory.relative_to(root)))
                latest,cumulative=next_history(cumulative,losses)
            block=dict(index=index,status="complete",infer_directory=str(inf.relative_to(root)),
                merge_directory=str(merge_dir.relative_to(root)),merged_checkpoint=str(merged),
                learner_directories=train_dirs,weights=weights,meta_losses=losses,training_parents=parents,
                bases_next=next_bases,latest_next=latest,cumulative_next=cumulative,
                checkpoint_fingerprints=[dict(path=p,files=checkpoint_manifest(p)) for p in sorted(set(next_bases))])
            state["blocks"].append(block);bases=next_bases
            turn_offset+=sum(len(q["turns"]) for q in chunk)
            state.update(status="running",pipeline_seconds=previous_seconds+time.perf_counter()-started)
            write_json(root/"state.json",state)
        state.update(status="complete",pipeline_seconds=previous_seconds+time.perf_counter()-started)
        write_json(root/"state.json",state)
        from .report import aggregate
        return aggregate(root)
    except Exception as exc:
        if state is not None:
            state.update(status="failed",error=str(exc),pipeline_seconds=previous_seconds+time.perf_counter()-started)
            write_json(root/"state.json",state)
        raise
    finally:
        lock.unlink(missing_ok=True)

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config");parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    for f in fields(Settings):
        kind=int if f.name=="order_seed" else type(f.default) if f.default is not MISSING else str
        flag="--"+f.name.replace("_","-")
        if kind is bool:
            parser.add_argument(flag,action=argparse.BooleanOptionalAction,default=None)
        else:
            parser.add_argument(flag,type=kind,default=None)
    args=vars(parser.parse_args(argv));cfg=args.pop("config");resume=args.pop("resume");dry=args.pop("dry_run")
    values=read_json(cfg) if cfg else {}
    values.update({k:v for k,v in args.items() if v is not None})
    try:
        return Settings(**values).validate(),resume,dry
    except (TypeError,ValueError) as exc:
        parser.error(str(exc))

def main():
    settings,resume,dry=parse_args();run(settings,resume,dry)
if __name__=="__main__":
    main()
