import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from transformers import LlamaConfig
from onlinespec_trace.core.backend.configs import EConfig
from onlinespec_trace.core.backend.cnets import Model
from onlinespec_trace.core.backend.modeling_llama_kv import LlamaForCausalLM
from onlinespec_trace.core.backend.ea_model_4 import EaModel

class Tokenizer:
    eos_token_id=None
    bos_token_id=1
    chat_template='test-only'
    def get_vocab(self): return {}
    def __call__(self,text,add_special_tokens=True):
        ids=[3+ord(x)%25 for x in text]
        return {'input_ids':([1] if add_special_tokens else [])+ids}
    def apply_chat_template(self,messages,**kwargs): return ' '.join(x['content'] for x in messages)+' A:'
    def decode(self,ids,**kwargs): return ''.join(chr(65+x%26) for x in ids)

@pytest.fixture
def tiny():
    torch.manual_seed(17);torch.set_num_threads(1)
    cfg=EConfig(vocab_size=32,draft_vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,rope_theta=10000.)
    draft=Model(cfg,load_emb=False,total_tokens=6,depth=2,top_k=2)
    draft.t2d.fill_(True);draft.d2t.zero_()
    target=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=8,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256))
    draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.data)
    return EaModel(target,draft,Tokenizer()),cfg.to_dict()
