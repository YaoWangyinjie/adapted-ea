import hashlib
import json
from pathlib import Path
import random
import os


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()


def write_json(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),"utf8")
    os.replace(temp,p)


def read_json(path): return json.loads(Path(path).read_text("utf8"))

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text("utf-8-sig").splitlines() if l.strip()]

def append_jsonl(handle, value):
    if isinstance(handle,(str,Path)):
        with Path(handle).open("a",encoding="utf8") as f: append_jsonl(f,value)
        return
    handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");handle.flush()


def write_jsonl(path, rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for r in rows: append_jsonl(f,r)


def load_questions(path, order_seed=None):
    rows=read_jsonl(path);seen=set()
    if not rows: raise ValueError("Empty dataset")
    for i,q in enumerate(rows):
        if "question_id" not in q or not isinstance(q.get("turns"),list) or not q["turns"]:
            raise ValueError(f"Expected question_id and nonempty turns list at row {i+1}")
        if not all(isinstance(t,str) and t.strip() for t in q["turns"]): raise ValueError("Invalid turn")
        key=str(q["question_id"])
        if key in seen: raise ValueError(f"Duplicate question_id {key}")
        seen.add(key);q["original_index"]=i
    if order_seed is not None: random.Random(order_seed).shuffle(rows)
    for i,q in enumerate(rows): q["stream_index"]=i
    return rows


def model_files(directory):
    p=Path(directory)
    if not p.is_dir(): raise ValueError(f"Use a downloaded local model directory: {p}")
    selected=sorted({*p.glob("*.safetensors"),*p.glob("pytorch_model*.bin"),*p.glob("*.json"),
                     *p.glob("tokenizer.model"),*p.glob("*.tiktoken")})
    return [{"name":f.name,"bytes":f.stat().st_size,"sha256":sha256(f)} for f in selected]
