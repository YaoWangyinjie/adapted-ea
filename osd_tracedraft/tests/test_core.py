import pytest
import torch
from osd_tracedraft.config import Settings
from osd_tracedraft.metrics import summarize,gains
from osd_tracedraft.checkpoints import validate_mapping,read_checkpoint,save_checkpoint
from osd_tracedraft.models import build_draft
from osd_tracedraft.training import recurrent_step,sequence_loss,teacher_example,train_block
from osd_tracedraft.io_utils import read_jsonl,token_hash
from osd_tracedraft.worker import infer_block,fit_prompt
from osd_tracedraft.run import parse_args
from osd_tracedraft.prompts import render_vicuna,VICUNA_SYSTEM

def settings(**kwargs):
    base=dict(model_profile='deepseek',method='osd_tracedraft',target='t',draft='d',questions='q',output='o',
              max_length=128,max_new_tokens=16,total_token=6,depth=2,top_k=2,dtype='float32',warmup_runs=0,
              osd_epochs=1,osd_batch_size=2,osd_unroll=7,trace_update_round=1,trace_lr=1e-4,
              osd_guard=False,trace_validation_guard=False,trace_diagnostics='full')
    base.update(kwargs);return Settings(**base).validate()

def test_metrics_count_pooling():
    rows=[dict(status='ok',total_accept_length=6,total_steps=2,total_drafted_tokens=10,new_tokens=7,
               generation_seconds=2,reset_seconds=.5),
          dict(status='ok',total_accept_length=2,total_steps=1,total_drafted_tokens=4,new_tokens=3,generation_seconds=1)]
    s=summarize(rows,training_seconds=1.5,pipeline_seconds=10)
    assert s['average_length']==8/3 and s['average_length_including_root']==11/3
    assert s['returned_tokens_per_step']==10/3 and s['discarded_verified_tokens']==1
    assert s['acceptance_percent']==800/14 and s['tokens_per_second_training_inclusive']==2
    assert gains(s,s)['speedup_training_inclusive']==1

def test_vicuna_prompt():
    assert render_vicuna([{'role':'user','content':'Hi'}])==VICUNA_SYSTEM+' USER: Hi ASSISTANT:'
    assert render_vicuna([{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'},
        {'role':'user','content':'Why?'}]).endswith('USER: Hi ASSISTANT: Hello</s>USER: Why? ASSISTANT:')

def test_cli_defaults():
    s,_,_=parse_args(['--model-profile','vicuna13b','--method','osd','--target','t','--draft','d',
                       '--questions','q','--output','o','--no-train-last-block'])
    assert s.max_length==2048 and s.osd_lr==1e-5 and s.osd_guard and s.trace_validation_guard and s.output_norm_fp32 and not s.train_last_block
    with pytest.raises(ValueError): settings(replay_mib=1)

def test_mapping():
    state={'t2d':torch.tensor([1,0,1,0],dtype=torch.bool),'d2t':torch.tensor([0,1])}
    validate_mapping(state,dict(vocab_size=4,draft_vocab_size=2))
    with pytest.raises(ValueError): validate_mapping({'d2t':torch.arange(4)},dict(vocab_size=4,draft_vocab_size=4))

def test_first_unroll_matches_inference(tiny):
    model,_=tiny;draft=model.ea_layer;draft.reset()
    ids=torch.tensor([[1,4,7,8,2]]);features=torch.randn(1,5,48)
    expected=draft(features,input_ids=ids)
    actual,_,_=recurrent_step(draft,draft.embed_tokens(ids),draft.fc(features),(),(),0)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

def test_checkpoint_recomputation_seven_step_gradients(tiny):
    model,_=tiny;d=model.ea_layer;d.requires_grad_(True);d.embed_tokens.requires_grad_(False)
    inputs=torch.tensor([[1,2,3,4,5,6,7,8,9]])
    features=torch.randn(1,9,48);probs=torch.randn(9,32).softmax(-1);valid=torch.tensor([0,0,1,1,1,1,1,1,0],dtype=torch.bool)
    loss,_=sequence_loss(d,inputs,features,probs,valid,7,.8,3,9,False);loss.backward()
    grads={k:p.grad.clone() for k,p in d.named_parameters() if p.requires_grad}
    expected=loss.detach();d.zero_grad(set_to_none=True)
    loss,logs=sequence_loss(d,inputs,features,probs,valid,7,.8,3,9,True);loss.backward()
    torch.testing.assert_close(loss,expected)
    for k,p in d.named_parameters():
        if p.requires_grad: torch.testing.assert_close(p.grad,grads[k],rtol=1e-5,atol=1e-6)
    assert len(logs)==7 and logs[-1]['positions']<logs[0]['positions']

def test_teacher_alignment(tiny):
    model,_=tiny;record=dict(input_ids=[1,3,4,5,6,7],loss_mask=[0,0,0,1,1,1])
    ids,features,probs,valid=teacher_example(model,record,2)
    assert ids.tolist()==[[3,4,5,6,7]] and valid.tolist()==[False,True,True,True,False]
    final,original=model._target(torch.tensor([record['input_ids']]))
    torch.testing.assert_close(features,original[:,:-1])
    torch.testing.assert_close(probs,model._head(final[:,1:])[0].float().softmax(-1))

def test_checkpoint_master_precision(tiny,tmp_path):
    model,raw=tiny
    save_checkpoint(tmp_path/'ckpt',model.ea_layer,raw)
    cfg,state=read_checkpoint(tmp_path/'ckpt')
    restored=build_draft(cfg,state,model.base_model.model.embed_tokens.weight,settings(dtype='float16'),torch.device('cpu'),training=True)
    torch.testing.assert_close(restored.lm_head.weight,model.ea_layer.lm_head.weight,rtol=0,atol=0)
    assert restored.lm_head.weight.dtype==torch.float32
    with pytest.raises(FileExistsError): save_checkpoint(tmp_path/'ckpt',model.ea_layer,raw)
    torch.save(state,tmp_path/'ckpt'/'pytorch_model.bin')
    with pytest.raises(ValueError,match='mixed'): read_checkpoint(tmp_path/'ckpt')

def test_osd_training_and_optimizer_resume(tiny,tmp_path):
    model,raw=tiny;draft=model.ea_layer
    record=dict(question_id='a',turn=0,input_ids=[1,3,4,5,6,7,8,9],loss_mask=[0,0,0,1,1,1,1,1])
    record['sequence_sha256']=token_hash(record['input_ids'])
    initial={k:p.clone() for k,p in draft.named_parameters()};target=[p.clone() for p in model.base_model.parameters()]
    report=train_block(model,raw,[record],settings(osd_lr=.0001),tmp_path/'initial',tmp_path/'next',tmp_path,0)
    assert report['steps']==1 and report['valid_positions']>0
    assert not torch.equal(initial['fc.weight'],draft.fc.weight)
    assert not torch.equal(initial['lm_head.weight'],draft.lm_head.weight)
    assert torch.equal(initial['embed_tokens.weight'],draft.embed_tokens.weight)
    assert all(torch.equal(a,b) for a,b in zip(target,model.base_model.parameters()))
    train_block(model,raw,[record],settings(osd_lr=.0001),tmp_path/'next',tmp_path/'next2',tmp_path,1)
    opt=torch.load(tmp_path/'next2'/'optimizer.pt',weights_only=True)
    assert all(v['step']==2 for v in opt['state'].values())

def test_real_inference_reset_and_records(tiny,tmp_path):
    model,_=tiny;s=settings(verify_greedy=1,verify_tokens=8)
    questions=[dict(question_id=1,stream_index=0,turns=['Hi','Why']),dict(question_id=2,stream_index=1,turns=['Test'])]
    report=infer_block(model,questions,s,dict(block_index=0,turn_offset=0),tmp_path)
    rows=read_jsonl(tmp_path/'turns.jsonl');records=read_jsonl(tmp_path/'training_records.jsonl')
    assert len(rows)==3 and report['summary']['turns']==3 and rows[0]['greedy_probe_passed'] is True
    assert rows[0]['adaptation']['status']=='updated'
    assert rows[1]['adaptation']['version_before']==1
    assert rows[2]['adaptation']['version_before']==0
    for row,record in zip(rows,records):
        assert sum(record['loss_mask'])==row['new_tokens']
        assert len(record['input_ids'])==row['prompt_tokens']+row['new_tokens']
        assert row['total_accept_length']+row['total_steps']>=row['new_tokens']

def test_prompt_truncation(tiny):
    model,_=tiny;s=settings(max_length=32,max_new_tokens=16)
    ids,removed=fit_prompt(list(range(1,40)),model.tokenizer,s)
    assert len(ids)==16 and ids[0]==1 and removed==23
    with pytest.raises(ValueError): fit_prompt(list(range(1,40)),model.tokenizer,settings(max_length=32,max_new_tokens=16,prompt_truncation='error'))


def test_block_persistent_replay_and_anchor_are_recorded(tiny,tmp_path):
    model,_=tiny
    cfg=settings(trace_mode='block_persistent',replay_mib=.125,replay_positions=4,anchor_weight=1e-4)
    infer_block(model,[dict(question_id=0,stream_index=0,turns=['Hi']),dict(question_id=1,stream_index=1,turns=['Test'])],
                cfg,dict(block_index=0,turn_offset=0),tmp_path)
    rows=read_jsonl(tmp_path/'trace_updates.jsonl')
    assert rows[1]['adaptation']['version_before']==1
    assert rows[1]['adaptation']['replay_positions']>0
    assert 0<rows[1]['state']['replay_bytes']<=int(.125*1024**2)
    assert rows[1]['adaptation']['config']['anchor_weight']==1e-4
    assert rows[1]['adaptation']['updates'][0]['anchor_loss']>0


def test_empty_supervision_cannot_decay_weights(tiny,tmp_path):
    model,raw=tiny
    record=dict(question_id='empty',turn=0,input_ids=[1,3,4,5],loss_mask=[0]*4)
    record['sequence_sha256']=token_hash(record['input_ids'])
    before={k:p.clone() for k,p in model.ea_layer.named_parameters()}
    with pytest.raises(ValueError,match='No covered assistant'):
        train_block(model,raw,[record],settings(),tmp_path/'in',tmp_path/'out',tmp_path,0)
    assert all(torch.equal(before[k],p) for k,p in model.ea_layer.named_parameters())
    assert not (tmp_path/'out').exists()


def test_metrics_reject_inconsistent_counts():
    with pytest.raises(ValueError,match='Inconsistent'):
        summarize([dict(status='ok',total_accept_length=3,total_steps=2,total_drafted_tokens=7,new_tokens=6,generation_seconds=1)])


def test_target_from_pretrained_and_saved_draft_round_trip(tiny,tmp_path):
    from osd_tracedraft.backend.modeling_llama_kv import LlamaForCausalLM
    from osd_tracedraft.backend.ea_model_4 import EaModel
    model,raw=tiny
    model.base_model.save_pretrained(tmp_path/'target',safe_serialization=True)
    save_checkpoint(tmp_path/'draft',model.ea_layer,raw)
    target=LlamaForCausalLM.from_pretrained(tmp_path/'target',torch_dtype=torch.float32,
                                          device_map={'':'cpu'},local_files_only=True)
    cfg,state=read_checkpoint(tmp_path/'draft')
    draft=build_draft(cfg,state,target.model.embed_tokens.weight,settings(),torch.device('cpu'))
    restored=EaModel(target,draft,model.tokenizer)
    prompt=torch.tensor([[1,3,5,7]])
    expected=model.eagenerate(prompt,max_new_tokens=12,max_length=64,enable_adaptation=False)
    actual=restored.eagenerate(prompt,max_new_tokens=12,max_length=64,enable_adaptation=False)
    assert torch.equal(expected,actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_draft_cuda_target_generation_and_teacher(tiny):
    model,_=tiny
    model.base_model.cuda()
    prompt=torch.tensor([[1,3,5,7]],device='cuda')
    expected=model.naivegenerate(prompt,max_new_tokens=12,max_length=64)
    model.setup_online_adaptation(learning_rate=1e-4,warmup_steps=1,update_at='warmup',validation_guard=False)
    actual,_,_,stats=model.eagenerate(prompt,max_new_tokens=12,max_length=64,log=True)
    assert torch.equal(actual,expected)
    assert stats['adaptation']['status']=='updated'
    record=dict(input_ids=[1,3,4,5,6,7],loss_mask=[0,0,0,1,1,1])
    ids,features,probs,valid=teacher_example(model,record,2)
    assert ids.device.type==features.device.type=='cpu'
    model.ea_layer.requires_grad_(True)
    model.ea_layer.embed_tokens.requires_grad_(False)
    loss,_=sequence_loss(model.ea_layer,ids,features,probs,valid,7,.8,2,5)
    loss.backward()
    assert model.ea_layer.fc.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_nf4_target_with_custom_kv_model(tiny,tmp_path):
    pytest.importorskip('bitsandbytes')
    from transformers import BitsAndBytesConfig
    from osd_tracedraft.backend.modeling_llama_kv import LlamaForCausalLM
    from osd_tracedraft.backend.ea_model_4 import EaModel
    model,_=tiny
    model.base_model.save_pretrained(tmp_path/'target',safe_serialization=True)
    target=LlamaForCausalLM.from_pretrained(tmp_path/'target',torch_dtype=torch.float16,
        device_map={'':'cuda:0'},local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    model.ea_layer.half()
    loaded=EaModel(target,model.ea_layer,model.tokenizer)
    prompt=torch.tensor([[1,3,5,7]],device='cuda')
    actual=loaded.eagenerate(prompt,max_new_tokens=8,max_length=64,enable_adaptation=False)
    assert actual.shape==(1,12)
    assert target.is_loaded_in_4bit
