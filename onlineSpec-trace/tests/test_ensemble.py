import ast
import copy
from pathlib import Path
from typing import List,Tuple
import pytest
import torch
import torch.nn.functional as F
from onlinespec_trace.config import Settings
from onlinespec_trace.ensemble import weights_for_block,next_history,merge_checkpoints,training_parents
from onlinespec_trace.training import upstream_meta_loss
from onlinespec_trace.core.checkpoints import save_checkpoint,read_checkpoint
from onlinespec_trace.core.io_utils import read_json

def settings(**kwargs):
    base=dict(model_profile="deepseek",target="t",draft="d",questions="q",output="o")
    base.update(kwargs)
    return Settings(**base).validate()

def upstream_function(filename,name):
    p=Path(__file__).resolve().parents[1]/"reference/EAGLE"/filename
    nodes=[x for x in ast.parse(p.read_text(encoding="utf8")).body if isinstance(x,ast.FunctionDef) and x.name==name]
    env=dict(torch=torch,F=F,List=List,Tuple=Tuple)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),"exec"),env)
    return env[name]

def test_meta_weights_match_upstream():
    losses=[.2,1.1,.5]
    hedge=upstream_function("pipeline_eagle3_hedge.py","compute_softmax_weights_from_losses")
    ens=upstream_function("pipeline_eagle3_ens.py","compute_weights_from_cumulative_losses")
    assert weights_for_block("hedge",losses,[10,20,30])==pytest.approx(hedge(losses,.2),abs=1e-7)
    assert weights_for_block("ens",losses,[10,20,30])==pytest.approx(ens([10,20,30],.2),abs=1e-7)
    assert weights_for_block("ens",None,[0,0,0])==[1/3]*3
    assert weights_for_block("hedge",None,[0,0,0])==[1/3]*3
    assert weights_for_block("ens",losses,[10,20,30],epsilon=0)==[1/3]*3

@pytest.mark.parametrize("values",[[float("nan"),1,2],[1,float("inf"),2],[-1,1,2],[1,2]])
def test_meta_rejects_bad_losses(values):
    with pytest.raises(ValueError):
        weights_for_block("hedge",values,[0,0,0])

def test_two_update_policies_are_distinct():
    latest,cumulative=next_history([10,0,0],[.1,1,.5])
    a=weights_for_block("hedge",latest,cumulative)
    b=weights_for_block("ens",latest,cumulative)
    assert a[0]==max(a) and b[0]==min(b)
    assert training_parents("hedge","merged",["a","b","c"])==["merged"]*3
    assert training_parents("ens","merged",["a","b","c"])==["a","b","c"]

def test_meta_loss_is_upstream_unweighted_epoch_batch_mean():
    rows=[
        dict(step=1,epoch=0,normalization=10,unroll=[[dict(ce_sum=10),dict(ce_sum=20)],[dict(ce_sum=20),dict(ce_sum=30)]]),
        dict(step=2,epoch=0,normalization=5,unroll=[[dict(ce_sum=5),dict(ce_sum=15)]]),
        dict(step=3,epoch=1,normalization=5,unroll=[[dict(ce_sum=10),dict(ce_sum=20)]])]
    result=upstream_meta_loss(rows)
    assert result["per_epoch_mean"]==[3.,3.] and result["total_loss"]==3.
    with pytest.raises(ValueError):
        upstream_meta_loss([])

@pytest.mark.parametrize("precision",["fp32","upstream_bf16"])
def test_merge_values_and_frozen_vocab_mapping(tiny,tmp_path,precision):
    model,raw=tiny;paths=[]
    for i in range(3):
        draft=copy.deepcopy(model.ea_layer)
        draft.lm_head.weight.data.fill_(i+1.)
        p=tmp_path/str(i);save_checkpoint(p,draft,raw);paths.append(p)
    merge_checkpoints(paths,[.2,.3,.5],tmp_path/"merged",precision)
    _,state=read_checkpoint(tmp_path/"merged")
    expected=torch.tensor(2.3) if precision=="fp32" else sum(w*torch.tensor(v,dtype=torch.bfloat16) for w,v in zip([.2,.3,.5],[1,2,3])).half()
    torch.testing.assert_close(state["lm_head.weight"],torch.full_like(state["lm_head.weight"],float(expected)))
    assert state["d2t"].dtype==torch.int64 and state["t2d"].dtype==torch.bool
    assert not (tmp_path/"merged/optimizer.pt").exists()
    with pytest.raises(FileExistsError):
        merge_checkpoints(paths,[1,1,1],tmp_path/"merged",precision)

def test_merge_refuses_different_mappings(tiny,tmp_path):
    model,raw=tiny;raw=dict(raw,vocab_size=64)
    from safetensors.torch import save_file
    from onlinespec_trace.core.io_utils import write_json
    state={k:v.clone() for k,v in model.ea_layer.state_dict().items() if k!="embed_tokens.weight"}
    paths=[]
    for i in range(3):
        p=tmp_path/str(i);p.mkdir()
        state["t2d"]=torch.arange(64)<32 if i<2 else torch.arange(64)>=32
        state["d2t"]=torch.zeros(32,dtype=torch.int64)+(0 if i<2 else 32)
        save_file(state,str(p/"model.safetensors"));write_json(p/"config.json",raw);paths.append(p)
    with pytest.raises(ValueError,match="mapping"):
        merge_checkpoints(paths,[1,1,1],tmp_path/"merged")

def test_runtime_configuration_keeps_outer_controls_matched():
    a=settings(method="onlinespec");b=settings(method="onlinespec_trace")
    for i in range(3):
        left=a.inner(i);right=b.inner(i)
        assert left.osd_lr==right.osd_lr==a.learning_rates[i]
        assert left.osd_weight_decay==0 and not left.osd_guard
        assert right.trace_validation_guard and right.has_trace and not left.has_trace
    assert settings(model_profile="vicuna13b").max_length==2048
    assert a.max_length==4096

@pytest.mark.parametrize("kw",[dict(variant="unknown"),dict(lr_1=0),dict(merge_precision="half"),dict(ens_epsilon=-1)])
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        settings(**kw)
