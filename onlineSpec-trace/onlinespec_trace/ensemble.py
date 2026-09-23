"""OnlineSPEC meta weights and parameter averaging (one serving draft)."""
import math
from pathlib import Path
import torch
from safetensors.torch import save_file
from .core.checkpoints import read_checkpoint,validate_mapping
from .core.io_utils import write_json,sha256

def weights_for_block(variant,latest,cumulative,temperature=0.2,epsilon=0.2):
    if len(cumulative)!=3 or (latest is not None and len(latest)!=3):
        raise ValueError("Exactly three learners are required")
    losses=cumulative if variant=="ens" else latest
    if variant not in ("ens","hedge"):
        raise ValueError("Unknown ensemble variant")
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(epsilon) or epsilon<0:
        raise ValueError("Invalid meta learner coefficient")
    if losses is None:
        return [1/3]*3
    if any(not math.isfinite(x) or x<0 for x in losses):
        raise ValueError("Nonfinite or negative meta loss")
    scale=epsilon if variant=="ens" else 1/temperature
    # Shift before scaling to avoid inf-inf for long cumulative histories.
    x=torch.tensor(losses,dtype=torch.float64)
    return torch.softmax(-(x-x.min())*scale,dim=0).tolist()

def next_history(cumulative,losses):
    if len(cumulative)!=3 or len(losses)!=3 or any(not math.isfinite(x) or x<0 for x in losses):
        raise ValueError("Invalid learner losses")
    values=[a+b for a,b in zip(cumulative,losses)]
    if any(not math.isfinite(x) for x in values):
        raise FloatingPointError("Cumulative loss overflow")
    return list(losses),values

def training_parents(variant,merged,bases):
    if len(bases)!=3 or variant not in ("hedge","ens"):
        raise ValueError("Invalid ensemble state")
    # The serving process's adapted head never appears in either branch.
    return [str(merged)]*3 if variant=="hedge" else list(bases)

def merge_checkpoints(paths,weights,destination,precision="fp32"):
    if len(paths)!=3 or len(weights)!=3 or precision not in ("fp32","upstream_bf16"):
        raise ValueError("Invalid merge configuration")
    if any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
        raise ValueError("Invalid merge weights")
    out=Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Never overwrite a merged checkpoint: {out}")
    weights=[w/sum(weights) for w in weights]
    merged={};config=None;originals={}
    dtype=torch.float32 if precision=="fp32" else torch.bfloat16
    for i,(path,w) in enumerate(zip(paths,weights)):
        cfg,state=read_checkpoint(path)
        if config is None:
            config=cfg
        elif cfg!=config or set(state)!=set(merged):
            raise ValueError("Learner config/state keys differ")
        for key,value in state.items():
            fixed=key in ("d2t","t2d","embed_tokens.weight") or not value.is_floating_point()
            if i==0:
                originals[key]=(value.shape,value.dtype)
                merged[key]=value.clone() if fixed else w*value.to(dtype)
            else:
                if value.shape!=originals[key][0]:
                    raise ValueError(f"Learner shape mismatch: {key}")
                if fixed:
                    if not torch.equal(merged[key],value):
                        raise ValueError(f"Frozen mapping/buffer differs: {key}")
                else:
                    merged[key]=merged[key]+w*value.to(dtype)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Nonfinite learner tensor: {key}")
        del state
    for key,value in merged.items():
        if key not in ("d2t","t2d","embed_tokens.weight") and value.is_floating_point():
            merged[key]=value.to(torch.float16 if precision=="upstream_bf16" else torch.float32)
            if not bool(torch.isfinite(merged[key]).all()):
                raise FloatingPointError(f"Nonfinite merged tensor: {key}")
    validate_mapping(merged,config)
    save_file({k:v.contiguous() for k,v in merged.items()},str(out/"model.safetensors"))
    write_json(out/"config.json",config)
    provenance=dict(parents=[str(Path(p).resolve()) for p in paths],weights=weights,
        precision=precision,weights_sha256=sha256(out/"model.safetensors"),
        optimizer="none; a parameter mixture has no inherited Adam state")
    write_json(out/"provenance.json",provenance)
    return provenance
