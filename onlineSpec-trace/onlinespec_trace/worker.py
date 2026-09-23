"""One target/draft per GPU process; each learner is trained sequentially."""
import argparse
import os
import platform
import random
import time
from pathlib import Path
import torch
import transformers
from .config import Settings
from .core.models import load_model
from .core.worker import infer_block
from .core.io_utils import read_json,read_jsonl,write_json,write_jsonl
from .training import train_learner

def execute(job):
    s=Settings(**job["settings"]).validate()
    directory=Path(job["directory"]);directory.mkdir(parents=True,exist_ok=True)
    if s.cpu_threads:
        torch.set_num_threads(s.cpu_threads)
    random.seed(s.seed);torch.manual_seed(s.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start=time.perf_counter()
    inner=s.inner(job.get("learner"))
    model,raw=load_model(inner,job["checkpoint_in"],training=job["stage"]=="train")
    model._sync();load_seconds=time.perf_counter()-start
    hardware=dict(gpu=torch.cuda.get_device_name(0),
        gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        python=platform.python_version(),draft_device=str(model.draft_device),
        cpu_threads=torch.get_num_threads(),dtype=s.dtype,
        output_norm_dtype=str(model.ea_layer.norm.weight.dtype),target_quantization=s.target_quantization)
    write_json(directory/"hardware.json",hardware)
    model._sync();start=time.perf_counter()
    if job["stage"]=="infer":
        result=infer_block(model,read_jsonl(job["questions"]),inner,job,directory)
        # Expose the correct experiment identity in every raw record.
        for name in ("turns","answers","training_records","trace_updates"):
            path=directory/(name+".jsonl")
            rows=read_jsonl(path)
            for row in rows:
                row["method"]=s.method
                row["ensemble_variant"]=s.variant
            write_jsonl(path,rows)
    elif job["stage"]=="train":
        result=train_learner(model,raw,read_jsonl(job["records"]),s,job,directory)
    else:
        raise ValueError("Unknown worker stage")
    model._sync()
    result.update(status="complete",stage_seconds=time.perf_counter()-start,load_seconds=load_seconds,
                  hardware=hardware,checkpoint_in=job["checkpoint_in"],block=job["block_index"])
    write_json(directory/"report.json",result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--job",required=True)
    args=p.parse_args();job=read_json(args.job)
    try:
        execute(job)
    except Exception as exc:
        write_json(Path(job["directory"])/"failure.json",dict(type=type(exc).__name__,message=str(exc)))
        raise

if __name__=="__main__":
    main()
