"""Reuse the verified EAGLE-3 recurrence; reproduce upstream meta-loss reduction."""
import math
from collections import defaultdict
from pathlib import Path
from .core.training import train_block
from .core.io_utils import read_jsonl,write_json

def upstream_meta_loss(updates):
    # traineagle3.py: average over epochs, positions (unrolls), and batches.
    # This is NOT the decayed objective used for backward or held-out loss.
    epochs=defaultdict(list)
    for row in updates:
        if "step" not in row:
            continue
        sequences=row["unroll"]
        lengths={len(s) for s in sequences}
        if len(lengths)!=1 or not lengths or not next(iter(lengths)):
            raise ValueError("Inconsistent unroll lengths in a training batch")
        count=next(iter(lengths))
        per_round=[sum(s[r]["ce_sum"] for s in sequences)/row["normalization"]
                   for r in range(count)]
        epochs[row["epoch"]].append(sum(per_round)/count)
    if not epochs:
        raise ValueError("No optimizer steps available for meta weighting")
    means=[sum(v)/len(v) for _,v in sorted(epochs.items())]
    result=sum(means)/len(means)
    if not math.isfinite(result) or result<0:
        raise FloatingPointError("Invalid OnlineSPEC meta loss")
    return dict(total_loss=result,per_epoch_mean=means,
                reduction="mean_epochs(mean_batches(mean_unroll(CE_sum / padded_batch_positions)))")

def train_learner(model,raw,records,settings,job,directory):
    index=job["learner"]
    inner=settings.inner(index)
    report=train_block(model,raw,records,inner,job["checkpoint_in"],
        job["checkpoint_out"],directory,job["block_index"])
    loss=upstream_meta_loss(read_jsonl(Path(directory)/"osd_updates.jsonl"))
    write_json(Path(directory)/"training_loss.json",loss)
    report.update(learner=index,learning_rate=settings.learning_rates[index],meta_loss=loss["total_loss"],
                  meta_loss_reduction=loss["reduction"])
    return report
