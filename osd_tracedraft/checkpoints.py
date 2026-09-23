"""Strict EAGLE-3 checkpoints. Original model directories are read-only."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file, save_file
from .io_utils import write_json, sha256


def read_state(directory):
    p=Path(directory)
    safe=list(p.glob('model*.safetensors'))
    binary=list(p.glob('pytorch_model*.bin'))
    if safe and binary:
        raise ValueError(f'{p}: mixed .bin/.safetensors; select one verified checkpoint in a clean directory')
    paths=sorted(safe or binary)
    if not paths: raise FileNotFoundError(f'No draft weights in {p}')
    state={}
    for path in paths:
        part=load_file(str(path),device='cpu') if safe else torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(part,dict) or any(not isinstance(v,torch.Tensor) for v in part.values()):
            raise ValueError(f'Expected a tensor state_dict: {path}')
        duplicates=set(state)&set(part)
        if duplicates: raise ValueError(f'Duplicate checkpoint keys: {duplicates}')
        state.update(part)
    return state


def validate_mapping(state, config):
    v=config['vocab_size'];d=config['draft_vocab_size']
    if d==v:
        # EAGLE's d2t is an OFFSET, not a token-index array.
        if 'd2t' in state and not torch.equal(state['d2t'].long(),torch.zeros(d,dtype=torch.long)):
            raise ValueError('Full-vocabulary d2t must be zero offsets')
        state['d2t']=torch.zeros(d,dtype=torch.long)
        state['t2d']=torch.ones(v,dtype=torch.bool)
    if 'd2t' not in state or 't2d' not in state: raise ValueError('Missing reduced-vocabulary mapping')
    mask=state['t2d'].bool();mapped=torch.arange(d)+state['d2t'].long()
    if mask.shape!=(v,) or int(mask.sum())!=d or not torch.equal(mapped,mask.nonzero().flatten()):
        raise ValueError('d2t/t2d mapping mismatch')
    state['d2t']=state['d2t'].long();state['t2d']=mask


def read_checkpoint(directory):
    config=json.loads((Path(directory)/'config.json').read_text(encoding='utf8'))
    if 'draft_vocab_size' not in config: raise ValueError('Use an EAGLE-3 draft, not EAGLE/EAGLE-2')
    state=read_state(directory)
    h=config.get('target_hidden_size',config['hidden_size'])
    if tuple(state.get('fc.weight',torch.empty(0)).shape)!=(config['hidden_size'],3*h):
        raise ValueError('Expected EAGLE-3 three-layer feature projection')
    validate_mapping(state,config)
    return config,state


def save_checkpoint(directory, draft, raw_config, optimizer=None, provenance=None):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    if any(p.glob('*.safetensors')) or any(p.glob('pytorch_model*.bin')):
        raise FileExistsError(f'Never overwrite a checkpoint: {p}')
    state={k:v.detach().cpu().contiguous() for k,v in draft.state_dict().items() if k!='embed_tokens.weight'}
    validate_mapping(state,raw_config)
    save_file(state,str(p/'model.safetensors'))
    write_json(p/'config.json',raw_config)
    if optimizer is not None: torch.save(optimizer.state_dict(),p/'optimizer.pt')
    write_json(p/'provenance.json',dict(provenance or {},weights_sha256=sha256(p/'model.safetensors')))


def copy_parent_checkpoint(parent, destination, provenance):
    """Publish an unchanged parent including its optimizer, never rejected state."""
    import shutil
    from .io_utils import token_hash
    source=Path(parent);out=Path(destination)
    if source.resolve()==out.resolve():
        raise ValueError('Checkpoint fallback must use a new directory')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite checkpoint: {out}')
    weights=sorted(source.glob('model*.safetensors')) or sorted(source.glob('pytorch_model*.bin'))
    if not weights or not (source/'config.json').exists():
        raise FileNotFoundError(f'Invalid fallback checkpoint: {source}')
    files=weights+[source/'config.json']
    if (source/'optimizer.pt').exists():
        files.append(source/'optimizer.pt')
    for path in files:
        shutil.copyfile(path,out/path.name)
    hashes=[dict(file=p.name,sha256=sha256(out/p.name)) for p in weights]
    digest=hashes[0]['sha256'] if len(hashes)==1 else token_hash(hashes)
    write_json(out/'provenance.json',dict(provenance,weights_sha256=digest,
        weight_files=hashes,unchanged_parent=True))
