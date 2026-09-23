"""OSD-EAGLE-3 block distillation with OnlineSPEC's seven-step recurrence.

Each recurrent position attends to the original causal prefix and its own
subsequent speculative states. Seven ordinary causal forwards are NOT equivalent.
The objective is masked soft cross entropy, sum_r decay**r * CE_r. Optimizer
batches are executed as single-sequence microbatches to bound GPU memory.
"""
import math
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from onlinespec_trace.core.backend.cnets import apply_rotary_pos_emb, repeat_kv
from .io_utils import append_jsonl, token_hash, write_json, read_jsonl
from .checkpoints import save_checkpoint, read_checkpoint, copy_parent_checkpoint


class NoTrainingPositions(ValueError):
    pass


def recurrent_step(draft,embedding,hidden,keys,values,step):
    layer=draft.midlayer;attn=layer.self_attn
    x=torch.cat((layer.input_layernorm(embedding),layer.hidden_norm(hidden)),dim=-1)
    b,n,_=x.shape;h=attn.num_heads;kv=attn.num_key_value_heads;d=attn.head_dim
    q=attn.q_proj(x).view(b,n,h,d).transpose(1,2)
    k=attn.k_proj(x).view(b,n,kv,d).transpose(1,2)
    v=attn.v_proj(x).view(b,n,kv,d).transpose(1,2)
    pos=torch.arange(n,device=x.device)[None]+step
    cos,sin=attn.rotary_emb(q,seq_len=n+step)
    q,k=apply_rotary_pos_emb(q,k,cos,sin,pos)
    k=repeat_kv(k,attn.num_key_value_groups);v=repeat_kv(v,attn.num_key_value_groups)
    all_k=tuple(keys)+(k,);all_v=tuple(values)+(v,)
    scores=q@all_k[0].transpose(-1,-2)/math.sqrt(d)
    causal=torch.ones(n,n,device=x.device,dtype=torch.bool).triu(1)
    scores=scores.masked_fill(causal,float('-inf'))
    if step:
        diag=torch.stack([(q*old).sum(-1)/math.sqrt(d) for old in all_k[1:]],dim=-1)
        scores=torch.cat((scores,diag),dim=-1)
    weights=scores.float().softmax(-1).to(q.dtype)
    out=weights[...,:n]@all_v[0]
    for i,old in enumerate(all_v[1:]): out=out+weights[...,n+i,None]*old
    out=attn.o_proj(out.transpose(1,2).reshape(b,n,-1))
    y=hidden+out
    return y+layer.mlp(layer.post_attention_layernorm(y)),k,v


@torch.no_grad()
def teacher_example(model,record,chunk_size):
    ids=torch.tensor([record['input_ids']],device=model.input_device,dtype=torch.long)
    if len(record['loss_mask'])!=ids.shape[1]: raise ValueError('Token/mask length mismatch')
    if ids.shape[1]<3: raise ValueError('Training sequence too short')
    model.base_model.model.tree_mask=None
    final,features=model._target(ids)
    mask=model.vocab_mask.to(final.device)
    probabilities=[];covered=[]
    for start in range(1,ids.shape[1],chunk_size):
        logits=model._head(final[:,start:start+chunk_size]).float()[0]
        covered.append(mask[logits.argmax(-1)].cpu())
        probabilities.append(logits[:,mask].softmax(-1).cpu())
    # Student row i consumes (H_i, token_(i+1)) and predicts token_(i+2).
    labels=torch.tensor(record['loss_mask'][2:]+[0],dtype=torch.bool)
    return ids[:,1:].to(model.draft_device),features[:,:-1].float(),torch.cat(probabilities),torch.cat(covered)&labels


def sequence_loss(draft,input_ids,features,probabilities,valid,unroll,decay,chunk_size,denominator,
                  checkpoint_graph=True):
    hidden=draft.fc(features);keys=();values=();n=hidden.shape[1]
    total=hidden.sum()*0;logs=[]
    for r in range(min(unroll,n)):
        shifted=torch.cat((input_ids[:,r:],torch.zeros_like(input_ids[:,:r])),dim=1) if r else input_ids
        with torch.no_grad(): emb=draft.embed_tokens(shifted).float()
        if checkpoint_graph:
            # Bind r: backward recomputation must not see the final loop index.
            def step_fn(e,z,*cache,r=r):
                return recurrent_step(draft,e,z,cache[:r],cache[r:],r)
            hidden,k,v=checkpoint(step_fn,emb,hidden,*keys,*values,use_reentrant=False)
        else: hidden,k,v=recurrent_step(draft,emb,hidden,keys,values,r)
        keys=keys+(k,);values=values+(v,)
        valid_r=torch.cat((valid[r:],torch.zeros(r,dtype=torch.bool))) if r else valid
        positions=valid_r.nonzero().flatten()
        loss_r=hidden.sum()*0
        for index in positions.split(chunk_size):
            if not len(index): continue
            x=hidden[0,index.to(hidden.device)]
            teacher=probabilities[index+r].to(hidden.device)
            def cross_entropy(z,t):
                return -(t*F.log_softmax(draft.lm_head(draft.norm(z)).float(),dim=-1)).sum()
            loss=checkpoint(cross_entropy,x,teacher,use_reentrant=False) if checkpoint_graph else cross_entropy(x,teacher)
            loss_r=loss_r+loss
        total=total+(decay**r)*loss_r/denominator
        logs.append({'unroll':r,'positions':len(positions),'ce_sum':float(loss_r.detach()),
                     'ce_per_valid_position':float(loss_r.detach())/len(positions) if len(positions) else None})
    return total,logs


def _optimize_block(model,raw_config,records,settings,checkpoint_in,checkpoint_out,log_dir,block_index):
    if not records: raise ValueError('No generated sequences to distill')
    for record in records:
        if len(record['input_ids'])>settings.max_length: raise ValueError('Unexpected overlength training record')
        if token_hash(record['input_ids'])!=record['sequence_sha256']: raise ValueError('Corrupted training sequence')
    draft=model.ea_layer
    # The target and token embedding remain frozen. Train all other draft weights.
    prepare_training_draft(model)
    params=[p for p in draft.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=settings.osd_lr,weight_decay=settings.osd_weight_decay)
    previous=Path(checkpoint_in)/'optimizer.pt'
    if previous.exists():
        optimizer.load_state_dict(torch.load(previous,map_location=model.draft_device,weights_only=True))
        for group in optimizer.param_groups: group.update(lr=settings.osd_lr,weight_decay=settings.osd_weight_decay)
    model._sync();start=time.perf_counter();steps=0;valid_positions=0
    for epoch in range(settings.osd_epochs):
        order=list(range(len(records)));random.Random(settings.seed+10007*block_index+epoch).shuffle(order)
        for offset in range(0,len(order),settings.osd_batch_size):
            batch=[records[i] for i in order[offset:offset+settings.osd_batch_size]]
            denominator=len(batch)*max(len(x['input_ids'])-1 for x in batch)
            optimizer.zero_grad(set_to_none=True);loss_value=0;logs=[]
            for record in batch:
                ids,features,probs,valid=teacher_example(model,record,settings.osd_loss_chunk)
                loss,per_round=sequence_loss(draft,ids,features,probs,valid,settings.osd_unroll,
                                             settings.osd_decay,settings.osd_loss_chunk,denominator)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite OSD loss')
                loss.backward();loss_value+=float(loss.detach());logs.append(per_round)
                valid_positions+=sum(x['positions'] for x in per_round)
                del loss,features,probs,ids
            if not any(item['positions'] for example in logs for item in example):
                append_jsonl(Path(log_dir)/'osd_updates.jsonl',dict(block=block_index,epoch=epoch,
                    status='skipped',reason='no_covered_assistant_positions',unroll=logs))
                continue
            norm=torch.nn.utils.clip_grad_norm_(params,settings.osd_clip,error_if_nonfinite=False)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite OSD gradient norm')
            optimizer.step();steps+=1
            if not all(bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Non-finite OSD weights')
            append_jsonl(Path(log_dir)/'osd_updates.jsonl',dict(block=block_index,epoch=epoch,step=steps,
                loss=loss_value,gradient_norm=float(norm),lr=settings.osd_lr,unroll=logs,
                record_ids=[(x['question_id'],x['turn']) for x in batch],normalization=denominator))
    model._sync();train_seconds=time.perf_counter()-start
    if not valid_positions: raise NoTrainingPositions('No covered assistant positions: refusing a meaningless update')
    return dict(optimization_seconds=train_seconds,steps=steps,valid_positions=valid_positions,
                trainable_parameters=sum(p.numel() for p in params),records=len(records)),optimizer


def split_validation_records(records, settings, block_index):
    """Hold out whole conversations, including every turn of each selected ID."""
    groups={}
    for record in records:
        key=token_hash(record['question_id'])
        groups.setdefault(key,[]).append(record)
    keys=sorted(groups)
    if len(keys)<2:
        return records,[]
    random.Random(settings.seed+10007*block_index+313).shuffle(keys)
    held=set(keys[:min(settings.osd_validation_questions,len(keys)-1)])
    return ([r for k,rs in groups.items() if k not in held for r in rs],
            [r for k,rs in groups.items() if k in held for r in rs])


def prepare_training_draft(model):
    draft=model.ea_layer
    # Frozen embedding lookup need not allocate a second FP32 vocabulary matrix.
    for name,child in draft.named_children():
        if name!='embed_tokens':
            child.float()
    draft.embed_tokens.to(dtype=model.base_model.model.embed_tokens.weight.dtype)
    draft.train().requires_grad_(True)
    draft.embed_tokens.requires_grad_(False)
    model.base_model.eval().requires_grad_(False)


@torch.no_grad()
def validation_metric(model, records, settings):
    """Same held-out sequences/positions before and after; FP32 draft metric."""
    was_training=model.ea_layer.training
    model.ea_layer.eval()
    numerator=denominator=0.0
    try:
        for record in records:
            ids,features,probs,valid=teacher_example(model,record,settings.osd_loss_chunk)
            positions=valid.nonzero().flatten()
            if len(positions)>settings.osd_validation_positions:
                pick=torch.linspace(0,len(positions)-1,settings.osd_validation_positions).long()
                valid=torch.zeros_like(valid);valid[positions[pick]]=True
            loss,logs=sequence_loss(model.ea_layer,ids,features,probs,valid,
                settings.osd_unroll,settings.osd_decay,settings.osd_loss_chunk,1,False)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite_osd_validation')
            numerator+=float(loss)
            denominator+=sum(settings.osd_decay**r['unroll']*r['positions'] for r in logs)
        if not denominator:
            raise FloatingPointError('no_osd_validation_positions')
        return dict(weighted_ce=numerator/denominator,weighted_positions=denominator,
                    records=len(records),dtype='float32')
    finally:
        model.ea_layer.train(was_training)


def train_block(model,raw_config,records,settings,checkpoint_in,checkpoint_out,log_dir,block_index):
    # Validate all records, including the held-out subset, before splitting.
    if not records:
        raise ValueError('No generated sequences to distill')
    for record in records:
        if len(record['input_ids'])>settings.max_length or token_hash(record['input_ids'])!=record['sequence_sha256']:
            raise ValueError('Corrupted or overlength training sequence')
    prepare_training_draft(model)
    train,held=(split_validation_records(records,settings,block_index)
                if settings.osd_guard else (records,[]))
    guard=dict(enabled=settings.osd_guard,accepted=None,
        training_ids=[dict(question_id=r['question_id'],turn=r['turn']) for r in train],
        validation_ids=[dict(question_id=r['question_id'],turn=r['turn']) for r in held],
        validation_before=None,validation_after=None,
        rtol=settings.osd_validation_rtol,atol=settings.osd_validation_atol)
    write_json(Path(log_dir)/'osd_validation_split.json',guard)
    started=time.perf_counter();optimizer=None;failure=None;report={}
    if settings.osd_guard and not held:
        failure='insufficient_conversations_for_holdout'
    try:
        if not failure:
            if settings.osd_guard:
                guard['validation_before']=validation_metric(model,held,settings)
            report,optimizer=_optimize_block(model,raw_config,train,settings,checkpoint_in,
                                            checkpoint_out,log_dir,block_index)
            # Prevent an FP32 checkpoint that overflows when loaded for inference.
            limit=torch.finfo(getattr(torch,settings.dtype)).max
            for name,p in model.ea_layer.named_parameters():
                if p.requires_grad and not (settings.output_norm_fp32 and name=='norm.weight'):
                    if float(p.detach().abs().max())>limit:
                        raise FloatingPointError('osd_inference_dtype_overflow')
            if settings.osd_guard:
                guard['validation_after']=validation_metric(model,held,settings)
                before=guard['validation_before']['weighted_ce']
                bound=before+settings.osd_validation_atol+settings.osd_validation_rtol*abs(before)
                guard['limit']=bound
                if guard['validation_after']['weighted_ce']>bound:
                    failure='osd_validation_regression'
    except (FloatingPointError, NoTrainingPositions) as exc:
        if not settings.osd_guard:
            raise
        failure=str(exc)
    # Count attempted updates even if optimization aborted before returning.
    attempts=read_jsonl(Path(log_dir)/'osd_updates.jsonl') if (Path(log_dir)/'osd_updates.jsonl').exists() else []
    attempted_steps=sum('step' in r for r in attempts)
    provenance=dict(parent=str(Path(checkpoint_in).resolve()),block=block_index,
        seed=settings.seed,method='OSD-EAGLE-3 block distillation',unroll=settings.osd_unroll,
        decay=settings.osd_decay,lr=settings.osd_lr,
        mask='new assistant tokens only; student (H_j, x_(j+1)) predicts x_(j+2)')
    if failure:
        guard.update(accepted=False,reason=failure)
        if optimizer is not None:
            optimizer.state.clear()
        optimizer=None
        model.ea_layer.zero_grad(set_to_none=True)
        # Restore the in-memory model as well as the on-disk continuation.
        _,state=read_checkpoint(checkpoint_in)
        loaded=model.ea_layer.load_state_dict(state,strict=False)
        if set(loaded.missing_keys)-{'embed_tokens.weight'} or loaded.unexpected_keys:
            raise ValueError('Cannot restore parent draft after guard rejection')
        del state
        copy_parent_checkpoint(checkpoint_in,checkpoint_out,dict(provenance,
            optimizer_steps=0,attempted_steps=attempted_steps,guard=guard))
        report.update(steps=0,records=len(train),valid_positions=report.get('valid_positions',0))
    else:
        guard['accepted']=True
        save_checkpoint(checkpoint_out,model.ea_layer,raw_config,optimizer,dict(provenance,
            optimizer_steps=report['steps'],attempted_steps=attempted_steps,guard=guard))
    write_json(Path(log_dir)/'osd_guard.json',guard)
    report.update(attempted_steps=attempted_steps,guard=guard,
        trainable_parameters=sum(p.numel() for p in model.ea_layer.parameters() if p.requires_grad),
        block_training_seconds=time.perf_counter()-started)
    return report
