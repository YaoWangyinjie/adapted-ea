import copy
import random
import pytest
import torch
from onlinespec_trace.core.backend.online_head import AdaptationConfig, OnlineHead, PositionReservoir
from onlinespec_trace.core.backend.cnets import LlamaRMSNorm

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

def test_fp32_norm_keeps_legacy_output_before_adaptation():
    torch.manual_seed(83)
    legacy=LlamaRMSNorm(32).half()
    legacy.weight.data.copy_(torch.randn(32).half())
    mixed=copy.deepcopy(legacy).float()
    x=torch.randn(50,32).half()
    torch.testing.assert_close(legacy(x),mixed(x),rtol=0,atol=0)
