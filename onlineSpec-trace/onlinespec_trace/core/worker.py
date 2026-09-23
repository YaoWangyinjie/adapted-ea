"""One process owns one stage on one visible GPU. Called by run.py."""
import argparse
import os
import platform
import random
import time
from pathlib import Path
import torch
import transformers
from onlinespec_trace.core.backend.online_head import AdaptationConfig
from .config import Settings
from .io_utils import read_json,read_jsonl,write_json,append_jsonl,token_hash
from .models import load_model
from .prompts import prompt_ids
from .metrics import summarize
from .training import train_block


def trace_config(s):
    return AdaptationConfig(learning_rate=s.trace_lr,source=s.trace_source,update_at='warmup',
        warmup_steps=s.trace_update_round,update_every=s.trace_every,
        max_prompt_positions=s.trace_prompt_positions,max_response_positions=s.trace_response_positions,
        replay_bytes=int(s.replay_mib*1024**2),replay_positions=s.replay_positions,
        anchor_weight=s.anchor_weight,seed=s.seed,
        validation_guard=s.trace_validation_guard,validation_rtol=s.trace_validation_rtol,
        validation_atol=s.trace_validation_atol,diagnostics=s.trace_diagnostics,
        diagnostics_every=s.trace_diagnostics_every)


def fit_prompt(ids,tokenizer,s):
    budget=s.max_length-s.max_new_tokens
    removed=max(0,len(ids)-budget)
    if removed:
        if s.prompt_truncation=='error': raise ValueError(f'Prompt has {len(ids)} tokens, budget {budget}')
        if ids[0]==tokenizer.bos_token_id: ids=ids[:1]+ids[-(budget-1):]
        else: ids=ids[-budget:]
    if len(ids)<2: raise ValueError('Prompt needs two tokens')
    return ids,removed


def infer_block(model,questions,s,job,directory):
    directory=Path(directory);rows=[];tok=model.tokenizer;llama3=s.model_profile=='deepseek'
    # Warm up CUDA without online learning. Restore scheduling afterwards.
    first=prompt_ids(tok,[{'role':'user','content':questions[0]['turns'][0]}],s.model_profile)
    first,_=fit_prompt(first,tok,s)
    ids=torch.tensor([first],device=model.input_device)
    model._sync();warm_start=time.perf_counter()
    for _ in range(s.warmup_runs):
        model.eagenerate(ids,max_new_tokens=min(32,s.max_new_tokens),max_length=s.max_length,
                          is_llama3=llama3,enable_adaptation=False)
    model._sync();warm_seconds=time.perf_counter()-warm_start
    if s.has_trace: model.setup_online_adaptation(trace_config(s))
    model.calls=job.get('turn_offset',0)
    write_json(directory/'adaptation_config.json',vars(model.adaptation_config) if model.adaptation_config else {})
    probe_seconds=0
    for question in questions:
        messages=[];reset_seconds=0
        if s.has_trace and s.trace_mode=='reset':
            model._sync();start=time.perf_counter();model.reset_online_adaptation();model._sync()
            reset_seconds=time.perf_counter()-start
        for turn,query in enumerate(question['turns']):
            messages.append({'role':'user','content':query})
            prompt=prompt_ids(tok,messages,s.model_profile)
            original_prompt_length=len(prompt);prompt,removed=fit_prompt(prompt,tok,s)
            ids=torch.tensor([prompt],device=model.input_device,dtype=torch.long)
            reference=None
            if question['stream_index']<s.verify_greedy:
                model._sync();start=time.perf_counter()
                reference=model.naivegenerate(ids,max_new_tokens=min(s.verify_tokens,s.max_new_tokens),
                                max_length=s.max_length,is_llama3=llama3)[0,len(prompt):].tolist()
                model._sync();probe_seconds+=time.perf_counter()-start
            out,n,_,stats=model.eagenerate(ids,max_new_tokens=s.max_new_tokens,max_length=s.max_length,
                         is_llama3=llama3,enable_adaptation=s.has_trace,log=True)
            generated=out[0,len(prompt):].tolist();sequence=prompt+generated
            if n!=len(generated): raise AssertionError('Returned token count mismatch')
            identity=dict(question_id=question['question_id'],stream_index=question['stream_index'],
                          turn=turn,block=job['block_index'],method=s.method)
            passed=reference is None or generated[:len(reference)]==reference
            row=dict(stats,**identity,status='ok' if passed else 'greedy_mismatch',
                reset_seconds=reset_seconds if turn==0 else 0,original_prompt_tokens=original_prompt_length,
                prompt_tokens=len(prompt),prompt_tokens_removed=removed,generated_sha256=token_hash(generated),
                prompt_sha256=token_hash(prompt),greedy_probe_passed=passed if reference is not None else None)
            append_jsonl(directory/'turns.jsonl',row);rows.append(row)
            text=tok.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            append_jsonl(directory/'answers.jsonl',dict(identity,answer=text,generated_ids=generated,
                            reference_probe_ids=reference,generated_sha256=token_hash(generated)))
            if not passed: raise RuntimeError('Greedy reference mismatch; inspect answers.jsonl before interpreting results')
            append_jsonl(directory/'trace_updates.jsonl',dict(identity,adaptation=stats['adaptation'],state=stats['adaptation_state']))
            append_jsonl(directory/'training_records.jsonl',dict(identity,input_ids=sequence,
                loss_mask=[0]*len(prompt)+[1]*len(generated),sequence_sha256=token_hash(sequence)))
            messages.append({'role':'assistant','content':text})
            print(f"{s.method} block={job['block_index']} q={question['stream_index']+1} turn={turn+1} tokens={n}",flush=True)
    return dict(summary=summarize(rows),warmup_seconds=warm_seconds,probe_seconds=probe_seconds)


def execute(job):
    s=Settings(**job['settings']).validate();directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
    if s.cpu_threads:
        torch.set_num_threads(s.cpu_threads)
    random.seed(s.seed);torch.manual_seed(s.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    start=time.perf_counter();model,raw=load_model(s,job['checkpoint_in'],training=job['stage']=='train');model._sync()
    load_seconds=time.perf_counter()-start
    hardware=dict(gpu=torch.cuda.get_device_name(0),gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,transformers=transformers.__version__,
        python=platform.python_version(),cuda=torch.version.cuda,dtype=s.dtype,
        target_quantization=s.target_quantization,draft_device=str(model.draft_device),
        cpu_threads=torch.get_num_threads(),output_norm_dtype=str(model.ea_layer.norm.weight.dtype))
    write_json(directory/'hardware.json',hardware)
    model._sync();stage_start=time.perf_counter()
    if job['stage']=='infer':
        result=infer_block(model,read_jsonl(job['questions']),s,job,directory)
    elif job['stage']=='train':
        result=train_block(model,raw,read_jsonl(job['records']),s,job['checkpoint_in'],job['checkpoint_out'],
                           directory,job['block_index'])
    else: raise ValueError('Unknown stage')
    model._sync();result.update(status='complete',load_seconds=load_seconds,stage_seconds=time.perf_counter()-stage_start,
        hardware=hardware,checkpoint_in=job['checkpoint_in'],block=job['block_index'])
    write_json(directory/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=read_json(args.job)
    try: execute(job)
    except Exception as exc:
        write_json(Path(job['directory'])/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise

if __name__=='__main__': main()
