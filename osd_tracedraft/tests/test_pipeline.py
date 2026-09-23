"""CPU integration across all four methods plus orchestration failure recovery."""
import copy
from pathlib import Path
import pytest
import torch
from osd_tracedraft.backend.ea_model_4 import EaModel
from osd_tracedraft import run as runner
from osd_tracedraft import queue as scheduler
from osd_tracedraft.models import build_draft,ModernRotary
from osd_tracedraft.checkpoints import read_checkpoint,save_checkpoint
from osd_tracedraft.training import train_block
from osd_tracedraft.worker import infer_block
from osd_tracedraft.io_utils import write_json,write_jsonl,read_json,read_jsonl,sha256
from osd_tracedraft.report import compare_runs
from osd_tracedraft.config import Settings
from test_core import settings


def test_all_four_methods_real_cpu_pipeline_and_resume(tiny,tmp_path,monkeypatch):
    original,raw=tiny
    draft_path=tmp_path/'draft';save_checkpoint(draft_path,original.ea_layer,raw)
    target_path=tmp_path/'target';target_path.mkdir();write_json(target_path/'config.json',{'tiny':True})
    questions=tmp_path/'questions.jsonl'
    write_jsonl(questions,[dict(question_id=i,turns=['A'+str(i)]) for i in range(4)])
    visited=[];fail_at={'enabled':False}
    def cpu_stage(job,root):
        visited.append((job['settings']['method'],job['block_index'],job['stage'],job['checkpoint_in']))
        if fail_at['enabled'] and job['block_index']==1 and job['stage']=='infer':
            fail_at['enabled']=False;raise RuntimeError('simulated failure between complete blocks')
        directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
        cfg=Settings(**job['settings']).validate();dc,state=read_checkpoint(job['checkpoint_in'])
        target=copy.deepcopy(original.base_model)
        draft=build_draft(dc,state,target.model.embed_tokens.weight,cfg,torch.device('cpu'),training=job['stage']=='train')
        model=EaModel(target,draft,original.tokenizer)
        if job['stage']=='infer':
            result=infer_block(model,read_jsonl(job['questions']),cfg,job,directory)
        else:
            result=train_block(model,dc,read_jsonl(job['records']),cfg,job['checkpoint_in'],job['checkpoint_out'],directory,job['block_index'])
        result.update(status='complete',stage_seconds=.25)
        write_json(directory/'report.json',result);return result
    monkeypatch.setattr(runner,'run_stage',cpu_stage)
    paths=[]
    for method in ('baseline','tracedraft','osd','osd_tracedraft'):
        path=tmp_path/method;paths.append(path)
        cfg=settings(method=method,target=str(target_path),draft=str(draft_path),questions=str(questions),
                     output=str(path),osd_block_size=2,osd_lr=.0001,verify_greedy=4,verify_tokens=8)
        if method=='osd_tracedraft':
            fail_at['enabled']=True
            with pytest.raises(RuntimeError,match='simulated failure'): runner.run(cfg)
            assert read_json(path/'state.json')['status']=='failed'
            before=len(visited);runner.run(cfg,resume=True)
            assert all(block==1 for _,block,_,_ in visited[before:])
        else: runner.run(cfg)
        complete_calls=len(visited);runner.run(cfg,resume=True);assert len(visited)==complete_calls
    rows=compare_runs(paths,tmp_path/'comparison')
    assert len(rows)==4 and rows[0]['speedup_generation']==1
    pair=compare_runs(paths[2:],tmp_path/'osd_pair',reference_method='osd')
    assert len(pair)==2 and pair[0]['speedup_generation']==1
    assert read_json(tmp_path/'osd_pair/comparison.json')['reference_method']=='osd'
    assert rows[2]['osd_training_seconds']==.25 and rows[3]['osd_training_seconds']==.25
    assert len({r['generated_tokens_sha256'] for r in rows})==1
    # The fast TraceDraft head must not leak into the outer OSD optimizer.
    assert sha256(paths[2]/'blocks/0000/train/checkpoint/model.safetensors')==sha256(paths[3]/'blocks/0000/train/checkpoint/model.safetensors')
    for method in ('osd','osd_tracedraft'):
        trained=[row for row in visited if row[0]==method and row[2]=='train']
        assert len(trained)==1 and Path(trained[0][3])==draft_path
    with pytest.raises(ValueError,match='settings differ'):
        runner.run(settings(**{**cfg.to_dict(),'trace_lr':.02}),resume=True)


def make_spec(tmp_path):
    target=tmp_path/'target';target.mkdir();draft=tmp_path/'draft';draft.mkdir()
    datasets=[]
    for name,count in [('large',5),('small',2)]:
        path=tmp_path/(name+'.jsonl');write_jsonl(path,[dict(question_id=i,turns=['hi']) for i in range(count)])
        datasets.append(dict(name=name,path=str(path),expected_questions=count))
    return dict(output_root=str(tmp_path/'results'),models=[dict(profile='deepseek',target=str(target),draft=str(draft))],datasets=datasets)


def test_queue_plan_order_and_count_validation(tmp_path):
    spec=make_spec(tmp_path);_,jobs=scheduler.build_jobs(spec)
    assert len(jobs)==8 and [x['question_count'] for x in jobs]==[2]*4+[5]*4
    spec['datasets'][0]['expected_questions']=500
    with pytest.raises(ValueError,match='expected 500'): scheduler.build_jobs(spec)


def test_queue_single_gpu_and_failure_continuation(tmp_path,monkeypatch):
    spec=make_spec(tmp_path);seen=[]
    class FakeProcess:
        def __init__(self,command,**kwargs):
            self.pid=100+len(seen);self.cfg=read_json(command[command.index('--config')+1])
            seen.append((self.cfg,kwargs['env']['CUDA_VISIBLE_DEVICES']))
        def wait(self): return 1 if self.cfg['method']=='tracedraft' else 0
    monkeypatch.setattr(scheduler.subprocess,'Popen',FakeProcess)
    with pytest.raises(RuntimeError,match='completion.json'): scheduler.run_queue(spec,['0','1','2','3'])
    assert len(seen)==8 and all(g in ('0','1','2','3') for _,g in seen)
    state=read_json(Path(spec['output_root'])/'queue/completion.json')
    assert len(state['jobs'])==8 and len(state['comparison_errors'])==2


def test_modern_rope_matches_transformers():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    cfg=LlamaConfig(hidden_size=32,num_attention_heads=4,max_position_embeddings=131072,rope_theta=500000.,
        rope_scaling={'rope_type':'llama3','factor':8.,'low_freq_factor':1.,'high_freq_factor':4.,'original_max_position_embeddings':8192})
    adapter=ModernRotary(cfg);official=LlamaRotaryEmbedding(cfg)
    x=torch.randn(1,4,12,8);a,b=adapter(x,seq_len=12);c,d=official(x,torch.arange(12)[None])
    torch.testing.assert_close(a[:,0],c);torch.testing.assert_close(b[:,0],d)
