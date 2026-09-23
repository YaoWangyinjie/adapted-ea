"""Optional numerical comparison with the inspected OnlineSPEC source checkout."""
import ast
import copy
import math
from pathlib import Path
from typing import List,Optional,Tuple,Union
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from osd_tracedraft.training import recurrent_step


def test_seven_step_recurrence_matches_onlinespec_source(tiny):
    source=Path(__file__).resolve().parents[3]/'OnlineSPEC/EAGLE/train_eagle3/cnets.py'
    if not source.exists(): pytest.skip('OnlineSPEC source checkout not included in portable package')
    wanted={'_make_causal_mask','_expand_mask','repeat_kv','rotate_half','apply_rotary_pos_emb',
            'LlamaRotaryEmbedding','LlamaLinearScalingRotaryEmbedding','LlamaDynamicNTKScalingRotaryEmbedding',
            'LlamaAttention','LlamaMLP','LlamaRMSNorm','LlamaDecoderLayeremb'}
    tree=ast.parse(source.read_text(encoding='utf8'))
    nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in wanted]
    namespace=dict(math=math,torch=torch,nn=nn,F=F,ACT2FN=ACT2FN,List=List,Optional=Optional,Tuple=Tuple,Union=Union)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    model,_=tiny;draft=model.ea_layer;draft.requires_grad_(True)
    upstream=namespace['LlamaDecoderLayeremb'](draft.config)
    upstream.load_state_dict(draft.midlayer.state_dict())
    a=torch.randn(1,9,16);b=a.clone();embedding=torch.randn(1,9,16)
    mask=torch.zeros(1,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),float('-inf'))
    keys=();values=();cache=[[],[]]
    for r in range(7):
        a,k,v=recurrent_step(draft,embedding,a,keys,values,r);keys=keys+(k,);values=values+(v,)
        out,cache=upstream(embedding,b,cache_hidden=cache,attention_mask=mask,position_ids=torch.arange(9)[None])
        b=out[0];torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    a.square().sum().backward();b.square().sum().backward()
    for (n,p),(m,q) in zip(draft.midlayer.named_parameters(),upstream.named_parameters()):
        assert n==m;torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=3e-5)
