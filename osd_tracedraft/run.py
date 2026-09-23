"""Run complete blockwise experiments; restart at completed block boundaries."""
import argparse
from dataclasses import fields,MISSING
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import Settings
from .io_utils import read_json,write_json,write_jsonl,load_questions,sha256,model_files

REPO=Path(__file__).resolve().parents[1]


def source_manifest():
    paths=sorted(list((REPO/'osd_tracedraft').glob('*.py'))+list((REPO/'osd_tracedraft/backend').glob('*.py')))
    return [{'name':str(p.relative_to(REPO)).replace('\\','/'),'sha256':sha256(p)} for p in paths]


def run_stage(job,root):
    directory=Path(job['directory']).resolve()
    if not directory.is_relative_to(Path(root).resolve()):
        raise ValueError('Stage directory must stay inside the run output directory')
    if directory.exists():
        archived=directory.with_name(directory.name+'.failed-'+str(time.time_ns()))
        directory.rename(archived)
    directory.mkdir(parents=True)
    path=directory/'job.json';write_json(path,job)
    # Unbuffered logs persist independently of this terminal or the agent.
    with (directory/'console.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,'-u','-m','osd_tracedraft.worker','--job',str(path)],cwd=REPO,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return read_json(directory/'report.json')


def run(settings,resume=False,dry_run=False):
    s=settings.validate()
    for key in ('target','draft','questions','output'): setattr(s,key,str(Path(getattr(s,key)).resolve()))
    questions=load_questions(s.questions,s.order_seed)
    chunks=[questions[i:i+s.osd_block_size] for i in range(0,len(questions),s.osd_block_size)]
    plan=dict(model=s.model_profile,method=s.method,questions=len(questions),turns=sum(len(q['turns']) for q in questions),
              blocks=len(chunks),training_blocks=(len(chunks) if s.train_last_block else max(0,len(chunks)-1)) if s.has_osd else 0,
              settings=s.to_dict())
    if dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    root=Path(s.output);root.mkdir(parents=True,exist_ok=True)
    lock=root/'RUNNING.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'{lock} exists: check whether a process is still running before removing a stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.perf_counter();state=None;previous_seconds=0
    try:
        if (root/'config.json').exists():
            if not resume: raise FileExistsError('Run exists. Use a new output or --resume with identical settings')
            if read_json(root/'config.json')!=s.to_dict(): raise ValueError('Resume settings differ')
            manifest=read_json(root/'manifest.json')
            if manifest['questions_sha256']!=sha256(s.questions) or manifest['source_files']!=source_manifest():
                raise ValueError('Data or source changed since run start')
            if manifest['target_files']!=model_files(s.target) or manifest['draft_files']!=model_files(s.draft):
                raise ValueError('Checkpoint changed since run start')
            state=read_json(root/'state.json');previous_seconds=state.get('pipeline_seconds',0)
            if state['status']=='complete':
                from .report import aggregate
                return aggregate(root)
        else:
            if any(p.name!='RUNNING.lock' for p in root.iterdir()): raise FileExistsError('Output directory is not empty')
            write_json(root/'config.json',s.to_dict());write_json(root/'plan.json',plan)
            write_jsonl(root/'question_order.jsonl',questions)
            write_json(root/'manifest.json',dict(questions_sha256=sha256(s.questions),target_files=model_files(s.target),
                draft_files=model_files(s.draft),source_files=source_manifest(),
                algorithm='OSD-EAGLE-3 block training + optional within-response TraceDraft',
                upstream_osd_commit='788a403d5495896b4fc5b7f56cfd41de5ae61967',
                upstream_onlinespec_commit='e58f82eb3f3adca3a686211236bf4f6e9e7e3a2b'))
            state=dict(status='running',blocks=[],pipeline_seconds=0)
            write_json(root/'state.json',state)
        checkpoint=s.draft;turn_offset=0
        for index,chunk in enumerate(chunks):
            completed=next((b for b in state['blocks'] if b['index']==index and b['status']=='complete'),None)
            if completed:
                checkpoint=completed['checkpoint_next'];turn_offset+=sum(len(q['turns']) for q in chunk);continue
            blockdir=root/'blocks'/f'{index:04d}';blockdir.mkdir(parents=True,exist_ok=True)
            qpath=blockdir/'questions.jsonl';write_jsonl(qpath,chunk)
            common=dict(settings=s.to_dict(),block_index=index,checkpoint_in=checkpoint,turn_offset=turn_offset)
            inf=blockdir/'infer'
            print(f"{s.method}: block {index+1}/{len(chunks)} inference",flush=True)
            run_stage(dict(common,stage='infer',directory=str(inf),questions=str(qpath)),root)
            train_dir=None;next_checkpoint=checkpoint
            if s.has_osd and (index<len(chunks)-1 or s.train_last_block):
                train_dir=blockdir/'train';next_checkpoint=str(train_dir/'checkpoint')
                print(f"{s.method}: block {index+1}/{len(chunks)} OSD training",flush=True)
                run_stage(dict(common,stage='train',directory=str(train_dir),records=str(inf/'training_records.jsonl'),
                               checkpoint_out=next_checkpoint),root)
            block=dict(index=index,status='complete',infer_directory=str(inf.relative_to(root)),
                       train_directory=str(train_dir.relative_to(root)) if train_dir else None,checkpoint_next=next_checkpoint)
            state['blocks']=[b for b in state['blocks'] if b['index']!=index]+[block]
            checkpoint=next_checkpoint;turn_offset+=sum(len(q['turns']) for q in chunk)
            state.update(status='running',pipeline_seconds=previous_seconds+time.perf_counter()-start)
            write_json(root/'state.json',state)
        state.update(status='complete',pipeline_seconds=previous_seconds+time.perf_counter()-start)
        write_json(root/'state.json',state)
        from .report import aggregate
        return aggregate(root)
    except Exception as exc:
        if state is not None:
            state.update(status='failed',error=str(exc),pipeline_seconds=previous_seconds+time.perf_counter()-start)
            write_json(root/'state.json',state)
        raise
    finally:
        lock.unlink(missing_ok=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='JSON containing Settings fields; CLI overrides it')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    for f in fields(Settings):
        kind=str if f.name=='order_seed' else type(f.default) if f.default is not MISSING else str
        if f.name=='order_seed': kind=int
        name='--'+f.name.replace('_','-')
        if kind is bool: parser.add_argument(name,action=argparse.BooleanOptionalAction,default=None)
        else: parser.add_argument(name,type=kind,default=None)
    args=vars(parser.parse_args(argv));cfg=args.pop('config');resume=args.pop('resume');dry=args.pop('dry_run')
    values=read_json(cfg) if cfg else {};values.update({k:v for k,v in args.items() if v is not None})
    try: s=Settings(**values).validate()
    except (TypeError,ValueError) as exc: parser.error(str(exc))
    return s,resume,dry


def main():
    s,resume,dry=parse_args();run(s,resume,dry)
if __name__=='__main__': main()
