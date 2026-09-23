import copy
import json
import random
from pathlib import Path
import pytest
import torch
from osd_tracedraft.backend.online_head import AdaptationConfig, OnlineHead, PositionReservoir
from osd_tracedraft.backend.cnets import LlamaRMSNorm
from osd_tracedraft.models import build_draft
from osd_tracedraft.checkpoints import save_checkpoint,read_checkpoint
from osd_tracedraft.io_utils import sha256,token_hash,read_jsonl
from osd_tracedraft import training
from test_core import settings


def rows_for(model,n=20):
    torch.manual_seed(41)
    return [dict(group='prompt' if i<n//2 else 'response',valid=True,
                 feature=torch.randn(16),target=torch.randn(32),draft_vocab_mass=1.,position=i+1)
            for i in range(n)]


def learner(model,**kwargs):
    return OnlineHead(model.ea_layer.norm,model.ea_layer.lm_head,
        AdaptationConfig(source='balanced',learning_rate=1e-4,max_step_drift=0,
                         replay_bytes=100000,replay_positions=4,**kwargs))


def test_fp32_norm_survives_sub_fp16_parameter_update():
    norm=LlamaRMSNorm(16)
    norm.weight.data.fill_(1.5)
    torch.manual_seed(7)
    x=torch.randn(64,16).half()
    before=norm(x)
    norm.weight.data.add_(1e-5)
    after=norm(x)
    assert norm.weight.dtype==torch.float32 and after.dtype==torch.float16
    assert torch.equal(norm.weight.half(),torch.full((16,),1.5,dtype=torch.float16))
    assert (before!=after).any()
    torch.nn.Linear(16,32,bias=False).half()(after)


def test_loaded_norm_is_not_rounded_before_fp32_restore(tiny):
    model,raw=tiny
    state={k:v.clone() for k,v in model.ea_layer.state_dict().items() if k!='embed_tokens.weight'}
    state['norm.weight'].fill_(1.0001)
    restored=build_draft(raw,state,model.base_model.model.embed_tokens.weight,
        settings(dtype='float16',output_norm_fp32=True),torch.device('cpu'))
    torch.testing.assert_close(restored.norm.weight,state['norm.weight'],rtol=0,atol=0)
    assert restored.lm_head.weight.dtype==torch.float16


def test_full_and_basic_diagnostics_do_not_change_updates(tiny):
    model,_=tiny;rows=rows_for(model)
    other=copy.deepcopy(model)
    a=learner(model,validation_guard=False,diagnostics='basic')
    b=learner(other,validation_guard=False,diagnostics='full')
    calls=[0,0]
    for index,obj in enumerate((a,b)):
        original=obj.evaluate
        def count(*args,_index=index,_original=original,**kwargs):
            calls[_index]+=1
            return _original(*args,**kwargs)
        obj.evaluate=count
    ra,rb=a.update(rows),b.update(rows)
    assert ra['status']==rb['status']=='updated'
    assert calls==[0,6]
    assert ra['train_before'] is None and rb['train_before'] is not None
    for name in a.master:
        torch.testing.assert_close(a.master[name],b.master[name],rtol=0,atol=0)


def test_guard_rolls_back_live_master_optimizer_replay_and_rng(tiny):
    model,_=tiny;rows=rows_for(model)
    obj=learner(model,validation_guard=False)
    assert obj.update(rows)['status']=='updated'
    obj.config.validation_guard=True
    master={k:v.clone() for k,v in obj.master.items()}
    live={k:v.clone() for k,v in obj.live.items()}
    moments={k:tuple(t.clone() for t in ts) for k,ts in obj.moments.items()}
    rng=obj.rng.getstate();history=list(obj.history);state=obj.state()
    calls=[]
    def metric(rows,parameters=None,inference=False):
        assert inference
        calls.append(1)
        return dict(kl=1.0 if len(calls)==1 else 2.0,top1_agreement=0.,positions=len(rows))
    obj.evaluate=metric
    result=obj.update(rows)
    assert result['status']=='rolled_back' and result['reason']=='validation_regression'
    assert result['inference_validation_after']['kl']==2.
    assert result['version_after']==state['weight_version']
    assert obj.rng.getstate()==rng and len(obj.history)==len(history)
    assert obj.history_bytes==state['replay_bytes'] and obj.step==state['optimizer_steps']
    for name in master:
        torch.testing.assert_close(obj.master[name],master[name],rtol=0,atol=0)
        torch.testing.assert_close(obj.live[name],live[name],rtol=0,atol=0)
    for name in moments:
        for before,after in zip(moments[name],obj.moments[name]):
            torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_validation_exception_restores_parameters(tiny):
    model,_=tiny;obj=learner(model);before={k:v.clone() for k,v in obj.live.items()}
    calls=[]
    def metric(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            raise RuntimeError('injected validation failure')
        return dict(kl=1.,positions=4,top1_agreement=0.)
    obj.evaluate=metric
    with pytest.raises(RuntimeError,match='injected'):
        obj.update(rows_for(model))
    assert obj.step==0 and obj.version==0 and not obj.moments
    for name in before:
        torch.testing.assert_close(obj.live[name],before[name],rtol=0,atol=0)


def test_missing_holdout_skips_without_advancing_state(tiny):
    model,_=tiny;obj=learner(model)
    result=obj.update(rows_for(model,4))
    assert result['status']=='skipped' and result['reason']=='no_validation_positions'
    assert obj.step==0 and not obj.moments and not obj.history


def test_basic_large_head_marks_sampled_statistics_explicitly():
    norm=LlamaRMSNorm(16);head=torch.nn.Linear(16,300,bias=False)
    obj=OnlineHead(norm,head,AdaptationConfig(source='prompt',validation_guard=False))
    rows=[dict(group='prompt',valid=True,feature=torch.randn(16),target=torch.randn(300),
               draft_vocab_mass=1.) for _ in range(8)]
    result=obj.update(rows);p=result['updates'][0]['parameters']['weight']
    assert p['inference_changed_fraction'] is None
    assert 0<=p['inference_changed_fraction_sampled']<=1
    assert 0<p['inference_sample_positions']<=4096


def test_prompt_presampling_matches_original_reservoir_decisions():
    cfg=AdaptationConfig(source='balanced',max_prompt_positions=4,max_response_positions=3)
    length=91;prompt=73;seed=5
    torch.manual_seed(4)
    logits=torch.randn(length,32)
    rng=random.Random(cfg.seed+104729*seed)
    expected={'prompt':[],'response':[]};seen={'prompt':0,'response':0}
    for i in range(1,length):
        group='prompt' if i<prompt-1 else 'response'
        seen[group]+=1;cap=getattr(cfg,'max_'+group+'_positions')
        slot=len(expected[group])
        if slot>=cap:
            slot=rng.randrange(seen[group])
            if slot>=cap:
                continue
        if slot==len(expected[group]): expected[group].append(i)
        else: expected[group][slot]=i
    batched=PositionReservoir(cfg,torch.ones(32,dtype=torch.bool),prompt,seed)
    for i in range(0,length,7): batched.observe(logits[i:i+7],i)
    prefill=PositionReservoir(cfg,torch.ones(32,dtype=torch.bool),prompt,seed)
    calls=[]
    def head(hidden):
        calls.append(hidden.shape[1])
        return hidden
    prefill.observe_prefill(logits[None],head)
    assert sum(calls)==sum(map(len,expected.values()))
    for obj in (batched,prefill):
        assert obj.seen==seen
        for group in expected:
            assert [r['position'] for r in obj.rows[group]]==expected[group]
            for row in obj.rows[group]:
                torch.testing.assert_close(row['target'],logits[row['position']])


def make_records():
    rows=[]
    for q in range(3):
        for turn in range(2):
            ids=[1,3+q,4,5+turn,6,7,8,9]
            rows.append(dict(question_id=q,turn=turn,input_ids=ids,loss_mask=[0,0,0,1,1,1,1,1],
                             sequence_sha256=token_hash(ids)))
    return rows


def save_parent(model,raw,path,optimizer=False):
    opt=None
    if optimizer:
        ps=[p for name,p in model.ea_layer.named_parameters() if name!='embed_tokens.weight']
        opt=torch.optim.AdamW(ps,lr=1e-5)
        for p in ps: p.grad=torch.zeros_like(p)
        opt.step();opt.zero_grad(set_to_none=True)
    save_checkpoint(path,model.ea_layer,raw,opt)


def test_osd_holdout_is_at_conversation_level():
    train,val=training.split_validation_records(make_records(),settings(osd_guard=True),2)
    assert {r['question_id'] for r in train}.isdisjoint({r['question_id'] for r in val})
    assert len(train)+len(val)==6 and len(train)==2
    assert training.split_validation_records(make_records(),settings(osd_guard=True),2)==(train,val)


def test_osd_rejection_restores_parent_and_optimizer(tiny,tmp_path,monkeypatch):
    model,raw=tiny;parent=tmp_path/'parent'
    save_parent(model,raw,parent,optimizer=True)
    before={k:p.clone() for k,p in model.ea_layer.named_parameters()}
    values=iter([1.,2.])
    monkeypatch.setattr(training,'validation_metric',lambda *args:dict(weighted_ce=next(values)))
    out=tmp_path/'out'
    result=training.train_block(model,raw,make_records(),settings(osd_guard=True,osd_epochs=1),
                               parent,out,tmp_path,0)
    assert result['steps']==0 and result['attempted_steps']==1 and not result['guard']['accepted']
    assert sha256(out/'model.safetensors')==sha256(parent/'model.safetensors')
    assert sha256(out/'optimizer.pt')==sha256(parent/'optimizer.pt')
    for k,p in model.ea_layer.named_parameters():
        torch.testing.assert_close(p,before[k],rtol=0,atol=0)
    trained={item[0] for row in read_jsonl(tmp_path/'osd_updates.jsonl') for item in row['record_ids']}
    held={r['question_id'] for r in result['guard']['validation_ids']}
    assert trained.isdisjoint(held)


def test_osd_acceptance_publishes_updated_checkpoint(tiny,tmp_path,monkeypatch):
    model,raw=tiny;parent=tmp_path/'parent';save_parent(model,raw,parent)
    values=iter([1.,.9])
    monkeypatch.setattr(training,'validation_metric',lambda *args:dict(weighted_ce=next(values)))
    out=tmp_path/'out'
    result=training.train_block(model,raw,make_records(),settings(osd_guard=True),
                               parent,out,tmp_path,0)
    assert result['steps']==1 and result['guard']['accepted']
    assert sha256(out/'model.safetensors')!=sha256(parent/'model.safetensors')
    opt=torch.load(out/'optimizer.pt',weights_only=True)
    assert all(v['step']==1 for v in opt['state'].values())


def test_osd_insufficient_groups_copies_parent_without_training(tiny,tmp_path):
    model,raw=tiny;parent=tmp_path/'parent';save_parent(model,raw,parent)
    result=training.train_block(model,raw,make_records()[:2],settings(osd_guard=True),
                               parent,tmp_path/'out',tmp_path,0)
    assert result['steps']==result['attempted_steps']==0
    assert result['guard']['reason']=='insufficient_conversations_for_holdout'
    assert sha256(parent/'model.safetensors')==sha256(tmp_path/'out/model.safetensors')


def test_osd_nonfinite_candidate_cannot_escape_guard(tiny,tmp_path,monkeypatch):
    model,raw=tiny;parent=tmp_path/'parent';save_parent(model,raw,parent)
    before=model.ea_layer.fc.weight.clone()
    monkeypatch.setattr(training,'validation_metric',lambda *args:dict(weighted_ce=1.))
    def fail(*args):
        with torch.no_grad(): model.ea_layer.fc.weight.fill_(float('nan'))
        raise FloatingPointError('injected_nonfinite')
    monkeypatch.setattr(training,'_optimize_block',fail)
    result=training.train_block(model,raw,make_records(),settings(osd_guard=True),
                               parent,tmp_path/'out',tmp_path,0)
    assert not result['guard']['accepted']
    torch.testing.assert_close(model.ea_layer.fc.weight,before,rtol=0,atol=0)


def test_osd_validation_metric_repeats_same_positions(tiny):
    model,_=tiny;cfg=settings(osd_guard=True,osd_validation_positions=3)
    training.prepare_training_draft(model)
    a=training.validation_metric(model,make_records()[:2],cfg)
    b=training.validation_metric(model,make_records()[:2],cfg)
    assert a==b and a['weighted_positions']>0 and a['dtype']=='float32'


@pytest.mark.parametrize('kwargs',[
    {'trace_diagnostics':'wrong'}, {'trace_validation_rtol':-1},
    {'osd_validation_atol':float('nan')}, {'osd_validation_questions':0}])
def test_guard_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)

def test_fp32_norm_keeps_legacy_output_before_adaptation():
    torch.manual_seed(83)
    legacy=LlamaRMSNorm(32).half()
    legacy.weight.data.copy_(torch.randn(32).half())
    mixed=copy.deepcopy(legacy).float()
    x=torch.randn(50,32).half()
    torch.testing.assert_close(legacy(x),mixed(x),rtol=0,atol=0)


def test_empty_osd_training_subset_falls_back(tiny,tmp_path,monkeypatch):
    model,raw=tiny;parent=tmp_path/'parent';save_parent(model,raw,parent)
    records=make_records()
    train,held=training.split_validation_records(records,settings(osd_guard=True),0)
    for r in train:
        r['loss_mask']=[0]*len(r['input_ids'])
    result=training.train_block(model,raw,records,settings(osd_guard=True),
                               parent,tmp_path/'out',tmp_path,0)
    assert not result['guard']['accepted'] and result['steps']==0
    assert sha256(parent/'model.safetensors')==sha256(tmp_path/'out/model.safetensors')
