"""Optional four-GPU scheduler: a matched pair occupies one GPU at a time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from queue import Queue,Empty
import re
import subprocess
import sys
from threading import Lock
from .config import Settings
from .core.io_utils import read_json,write_json,load_questions

PROJECT=Path(__file__).resolve().parents[1]

def build_jobs(spec):
    jobs=[]
    for model in spec["models"]:
        for dataset in spec["datasets"]:
            for name in (model["name"],dataset["name"]):
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name):
                    raise ValueError("Use simple model/dataset names without path separators")
            questions=load_questions(dataset["path"])
            if dataset.get("expected_questions") is not None and len(questions)!=dataset["expected_questions"]:
                raise ValueError(f"Wrong question count for {dataset['name']}")
            for seed in spec.get("seeds",[0]):
                root=Path(spec["output_root"]).resolve()/model["name"]/dataset["name"]/f"seed_{seed}"
                values=dict(spec.get("settings",{}),model_profile=model["profile"],target=model["target"],
                    draft=model["draft"],questions=dataset["path"],output=str(root),seed=seed)
                settings=Settings(**values).validate()
                for key in ("target","draft","questions"):
                    setattr(settings,key,str(Path(getattr(settings,key)).resolve()))
                jobs.append(dict(name=f"{model['name']}/{dataset['name']}/seed_{seed}",
                    questions=len(questions),turns=sum(len(q["turns"]) for q in questions),settings=settings.to_dict()))
    names=[j["name"] for j in jobs]
    if len(names)!=len(set(names)):
        raise ValueError("Duplicate queue task names")
    return sorted(jobs,key=lambda j:(j["questions"],j["turns"],j["name"]))

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--gpus",default="0,1,2,3")
    p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    gpus=a.gpus.split(",")
    if not gpus or len(gpus)!=len(set(gpus)) or any(not x.isdigit() for x in gpus):
        p.error("Provide distinct GPU indices, e.g. 0,1,2,3")
    spec=read_json(a.config);jobs=build_jobs(spec)
    if a.dry_run:
        print(__import__("json").dumps(dict(pairs=len(jobs),runs=len(jobs)*2,jobs=jobs),ensure_ascii=False,indent=2))
        return
    root=Path(spec["output_root"]).resolve()/"queue";root.mkdir(parents=True,exist_ok=True)
    plan=root/"plan.json"
    if plan.exists() and read_json(plan)!=jobs:
        raise ValueError("Queue plan changed; use a new output root")
    write_json(plan,jobs)
    pending=Queue()
    for i,job in enumerate(jobs):
        pending.put((i,job))
    statuses={};lock=Lock()
    def record(i,**values):
        with lock:
            statuses[str(i)]=values
            write_json(root/"status.json",statuses)
    def worker(gpu):
        while True:
            try:
                i,job=pending.get_nowait()
            except Empty:
                return
            cfg=root/f"job_{i:04d}.json";write_json(cfg,job["settings"])
            env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=gpu
            cmd=[sys.executable,"-u","-m","onlinespec_trace.pair","--config",str(cfg)]
            if a.resume:
                cmd.append("--resume")
            with (root/f"job_{i:04d}.log").open("a",encoding="utf8") as log:
                process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                record(i,name=job["name"],gpu=gpu,pid=process.pid,status="running")
                code=process.wait()
            record(i,name=job["name"],gpu=gpu,status="complete" if code==0 else "failed",exit_code=code)
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,gpus))
    failed=[v for v in statuses.values() if v["status"]!="complete"]
    write_json(root/"completion.json",dict(pairs=len(jobs),failed=failed,results=statuses))
    if failed:
        raise SystemExit(1)
if __name__=="__main__":
    main()
