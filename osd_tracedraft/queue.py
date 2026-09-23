"""A local four-worker GPU queue; no polling agent or scheduled automation."""
import argparse
import concurrent.futures
from dataclasses import fields
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from .config import Settings,METHODS
from .io_utils import load_questions,read_json,write_json
from .run import REPO


def slug(text):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',text) or text in ('.','..'): raise ValueError(f'Unsafe job label: {text}')
    return text


def build_jobs(spec):
    root=Path(spec['output_root']).expanduser().resolve();jobs=[]
    methods=spec.get('methods',list(METHODS));seeds=spec.get('seeds',[0])
    if 'baseline' not in methods: raise ValueError('Include baseline for matched speedup')
    if len(methods)!=len(set(methods)) or len(seeds)!=len(set(seeds)): raise ValueError('Duplicate methods/seeds')
    for model in spec['models']:
        name=slug(model.get('name',model['profile']))
        for dataset in spec['datasets']:
            label=slug(dataset['name']);path=Path(dataset['path']).expanduser().resolve()
            questions=load_questions(path)
            expected=dataset.get('expected_questions')
            if expected is not None and len(questions)!=expected:
                raise ValueError(f'{label}: expected {expected} questions, found {len(questions)}')
            for seed in seeds:
                group=root/name/label/f'seed_{seed}'
                for method in methods:
                    values=dict(spec.get('settings',{}));values.update(model.get('settings',{}));values.update(dataset.get('settings',{}))
                    values.update(model_profile=model['profile'],method=method,target=str(Path(model['target']).expanduser().resolve()),
                        draft=str(Path(model['draft']).expanduser().resolve()),questions=str(path),output=str(group/method),seed=seed)
                    cfg=Settings(**values).validate()
                    if not Path(cfg.target).is_dir() or not Path(cfg.draft).is_dir(): raise FileNotFoundError(f'Model paths missing for {name}')
                    jobs.append(dict(label=f'{name}/{label}/seed_{seed}/{method}',settings=cfg.to_dict(),group=str(group),
                        question_count=len(questions),turn_count=sum(len(q['turns']) for q in questions)))
    jobs.sort(key=lambda j:(j['question_count'],j['turn_count'],j['label']))
    if len({j['settings']['output'] for j in jobs})!=len(jobs): raise ValueError('Duplicate output directories')
    return root,jobs


def run_queue(spec,gpus,resume=False,dry_run=False):
    if not gpus or len(set(gpus))!=len(gpus) or any(not str(g).isdigit() for g in gpus): raise ValueError('Use distinct GPU indices')
    root,jobs=build_jobs(spec)
    if dry_run:
        for j in jobs: print(f"{j['question_count']:>5} questions {j['turn_count']:>5} turns  {j['label']}")
        print(f'{len(jobs)} jobs on {len(gpus)} independent single-GPU workers');return
    q=queue.Queue();[q.put(j) for j in jobs]
    control=root/'queue';control.mkdir(parents=True,exist_ok=True)
    write_json(control/'plan.json',dict(gpus=gpus,jobs=jobs))
    lock=threading.Lock();outcomes=[];active={}
    def save_status(): write_json(control/'status.json',dict(active=active,finished=outcomes,remaining=q.qsize()))
    def worker(gpu):
        while True:
            try: job=q.get_nowait()
            except queue.Empty: return
            tag=job['label'].replace('/','__');cfgpath=control/(tag+'.json');write_json(cfgpath,job['settings'])
            command=[sys.executable,'-u','-m','osd_tracedraft.run','--config',str(cfgpath)]
            if resume: command.append('--resume')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            start=time.time()
            with (control/(tag+'.log')).open('w',encoding='utf8') as log:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                with lock:
                    active[str(gpu)]=dict(job=job['label'],pid=process.pid,started=start);save_status()
                    print(f"GPU {gpu} started {job['label']} (pid {process.pid})",flush=True)
                code=process.wait() # OS wait, no model tokens or periodic monitoring
            with lock:
                outcomes.append(dict(job=job['label'],gpu=gpu,exit_code=code,seconds=time.time()-start,
                                     output=job['settings']['output']))
                active.pop(str(gpu),None);save_status()
                print(f"GPU {gpu} finished {job['label']}: exit={code}",flush=True)
            q.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for f in futures: f.result()
    from .report import compare_runs
    errors=[]
    for group in sorted({j['group'] for j in jobs}):
        paths=[j['settings']['output'] for j in jobs if j['group']==group]
        failed=[x for x in outcomes if x['output'] in paths and x['exit_code']]
        if failed: errors.append(dict(group=group,error='one or more jobs failed'));continue
        try: compare_runs(paths,Path(group)/'comparison')
        except Exception as exc: errors.append(dict(group=group,error=str(exc)))
    write_json(control/'completion.json',dict(jobs=outcomes,comparison_errors=errors))
    if errors or any(x['exit_code'] for x in outcomes): raise RuntimeError(f'See {control}/completion.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    run_queue(read_json(a.config),a.gpus.split(','),a.resume,a.dry_run)
if __name__=='__main__': main()
