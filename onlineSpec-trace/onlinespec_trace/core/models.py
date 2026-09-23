"""Both profiles use the local Llama KV implementation; no auto device sharding."""
import os
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from onlinespec_trace.core.backend.modeling_llama_kv import LlamaForCausalLM
from onlinespec_trace.core.backend.configs import EConfig
from onlinespec_trace.core.backend.cnets import Model
from onlinespec_trace.core.backend.ea_model_4 import EaModel
from .checkpoints import read_checkpoint
from .config import PROFILES


class ModernRotary(nn.Module):
    """Adapter from Transformers RoPE parameters to cnets' cos/sin interface."""
    def __init__(self,config):
        super().__init__()
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        self.config=config
        kind=(config.rope_scaling or {}).get('rope_type',(config.rope_scaling or {}).get('type','default'))
        self.fn=ROPE_INIT_FUNCTIONS[kind]
        inv,scale=self.fn(config,device=torch.device('cpu'))
        self.register_buffer('inv_freq',inv,persistent=False);self.scale=scale
        self.kind=kind
    def forward(self,x,seq_len=None):
        inv=self.inv_freq.float().to(x.device);scale=self.scale
        if self.kind=='dynamic' and seq_len>self.config.max_position_embeddings:
            inv,scale=self.fn(self.config,device=x.device,seq_len=seq_len)
        t=torch.arange(seq_len,device=x.device,dtype=torch.float32)
        phase=torch.outer(t,inv);phase=torch.cat((phase,phase),dim=-1)
        return (phase.cos()*scale)[None,None].to(x.dtype),(phase.sin()*scale)[None,None].to(x.dtype)


def build_draft(raw,state,target_embedding,settings,device,training=False):
    # EConfig validates only legacy RoPE. Preserve the original JSON and install
    # a compatible rotary module for modern scaling without changing its values.
    clean=dict(raw);scaling=clean.pop('rope_scaling',None)
    cfg=EConfig(**clean,rope_scaling=None)
    draft=Model(cfg,load_emb=False,total_tokens=settings.total_token,depth=settings.depth,top_k=settings.top_k)
    cfg.rope_scaling=scaling
    if scaling:
        draft.midlayer.self_attn.rotary_emb=ModernRotary(cfg)
    if tuple(target_embedding.shape)!=tuple(draft.embed_tokens.weight.shape):
        raise ValueError('Draft embedding width/vocabulary must match the selected target')
    if 'embed_tokens.weight' in state:
        if not torch.equal(state['embed_tokens.weight'].to(target_embedding.dtype),target_embedding.detach().cpu()):
            raise ValueError('Draft embedding differs from target; unsupported checkpoint')
    draft.embed_tokens.weight.data.copy_(target_embedding.detach().cpu())
    loaded=draft.load_state_dict(state,strict=False)
    missing=set(loaded.missing_keys)-{'embed_tokens.weight'}
    if missing or loaded.unexpected_keys:
        raise ValueError(f'Incompatible draft: missing={missing}, unexpected={loaded.unexpected_keys}')
    draft=draft.to(device=device,dtype=torch.float32 if training else getattr(torch,settings.dtype))
    if settings.output_norm_fp32:
        draft.norm.float()
        # Restore directly from the checkpoint, not from the rounded FP16 copy.
        draft.norm.weight.data.copy_(state['norm.weight'].to(device=device,dtype=torch.float32))
    return draft


def load_model(settings,checkpoint,training=False):
    if not torch.cuda.is_available(): raise RuntimeError('Production experiments require one CUDA GPU')
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0 (or 1/2/3)')
    if settings.dtype=='bfloat16' and not torch.cuda.is_bf16_supported(): raise ValueError('BF16 unsupported')
    if not Path(settings.target).is_dir(): raise FileNotFoundError('Provide a local target checkpoint directory')
    cfg=AutoConfig.from_pretrained(settings.target,local_files_only=True)
    expected=PROFILES[settings.model_profile]
    actual=(cfg.hidden_size,cfg.num_hidden_layers,cfg.vocab_size)
    wanted=(expected['hidden_size'],expected['layers'],expected['vocab_size'])
    if cfg.model_type!='llama' or actual!=wanted:
        raise ValueError(f'Target profile mismatch: {actual} != {wanted}')
    if settings.max_length>cfg.max_position_embeddings:
        raise ValueError('Requested context exceeds target config; do not silently extend Vicuna RoPE')
    options={}
    if settings.target_quantization=='nf4':
        from transformers import BitsAndBytesConfig
        options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch,settings.dtype))
    target=LlamaForCausalLM.from_pretrained(settings.target,torch_dtype=getattr(torch,settings.dtype),
            device_map={'':'cuda:0'},local_files_only=True,**options).eval()
    raw,state=read_checkpoint(checkpoint)
    if raw['vocab_size']!=cfg.vocab_size: raise ValueError('Target/draft vocabulary mismatch')
    device=torch.device('cuda:0' if settings.draft_device=='cuda' else 'cpu')
    draft=build_draft(raw,state,target.model.embed_tokens.weight,settings,device,training=training)
    del state
    tok=AutoTokenizer.from_pretrained(settings.target,use_fast=False,local_files_only=True)
    if settings.model_profile=='deepseek' and not tok.chat_template:
        raise ValueError('DeepSeek tokenizer must contain its official chat template')
    model=EaModel(target,draft,tok)
    model.ea_layer.reset();model.base_model.model.tree_mask=None
    return model,raw
