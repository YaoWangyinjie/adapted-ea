"""Diagnose a full-output mismatch without retraining or modifying run records."""
from pathlib import Path
import json
import torch
from onlinespec_trace.config import Settings
from onlinespec_trace.core.models import load_model
from onlinespec_trace.core.worker import trace_config
from onlinespec_trace.core.io_utils import read_json,read_jsonl,write_json
from onlinespec_trace.core.backend import utils

ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/"validation/numerical_diagnostic"
    out.mkdir(exist_ok=True)
    if (out/"report.json").exists():
        raise FileExistsError("Use an independent diagnostic directory")
    run=ROOT/"validation/real_8b"
    s=Settings(**read_json(run/"onlinespec/config.json")).validate()
    s.method="onlinespec_trace"
    torch.set_num_threads(s.cpu_threads);torch.manual_seed(s.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    records=read_jsonl(run/"onlinespec/blocks/0001/infer/training_records.jsonl")
    record=records[0];prompt_length=record["loss_mask"].index(1)
    prompt=record["input_ids"][:prompt_length]
    expected=record["input_ids"][prompt_length:]
    model,_=load_model(s.inner(),run/"onlinespec/blocks/0001/merge/checkpoint")
    ids=torch.tensor([prompt],device=model.input_device)
    model.eagenerate(ids,max_new_tokens=32,max_length=s.max_length,enable_adaptation=False)
    reference=model.naivegenerate(ids,max_new_tokens=32,max_length=s.max_length)[0,prompt_length:].tolist()
    original=utils.evaluate_posterior
    results=[]
    for method in ("onlinespec","onlinespec_trace"):
        if method=="onlinespec_trace":
            model.setup_online_adaptation(trace_config(s.inner()))
            model.reset_online_adaptation()
        consumed=[];observations=[]
        def audited(logits,candidates,processor):
            best,length,next_logits=original(logits,candidates,processor)
            best,length=int(best),int(length)
            count=min(length+1,32-len(consumed))
            path=candidates[best,:count].tolist()
            for j in range(count):
                next_index=len(consumed)+j+1
                if next_index==29:
                    row=logits[best,j]
                    observations.append(dict(next_index=next_index,
                        prefix_ids=consumed+path[:j+1],matches_shared_prefix=consumed+path[:j+1]==expected[:29],
                        and_logit=float(row[323]),but_logit=float(row[719]),argmax=int(row.argmax()),
                        top_ids=row.topk(5).indices.tolist(),top_logits=row.topk(5).values.tolist()))
            consumed.extend(path)
            return best,torch.tensor(length,device=logits.device),next_logits
        utils.evaluate_posterior=audited
        model.calls=4
        try:
            output,_,_,stats=model.eagenerate(ids,max_new_tokens=32,max_length=s.max_length,
                enable_adaptation=method=="onlinespec_trace",log=True)
        finally:
            utils.evaluate_posterior=original
        results.append(dict(method=method,generated=output[0,prompt_length:].tolist(),
            observations=observations,adaptation=stats["adaptation"]))
    prefix=torch.tensor([prompt+expected[:29]],device=model.input_device)
    model.base_model.model.tree_mask=None
    with torch.no_grad():
        final=model.base_model.model(input_ids=prefix,use_cache=False,return_dict=True).last_hidden_state
        logits=model._head(final[:,-1:])[0,-1]
    report=dict(question_id=record["question_id"],turn=0,target_only_reference=reference,
        records=results,canonical_shared_prefix=dict(and_logit=float(logits[323]),but_logit=float(logits[719]),
        argmax=int(logits.argmax()),top_ids=logits.topk(5).indices.tolist(),top_logits=logits.topk(5).values.tolist()))
    write_json(out/"report.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
