"""Pooled count metrics; never average rounded acceptance percentages."""
import math

def validate_counts(row):
    names=('total_accept_length','total_steps','total_drafted_tokens','new_tokens')
    counts=[row.get(k,0) for k in names]
    if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in counts):
        raise ValueError('Token/step counters must be nonnegative integers')
    a,s,w,n=counts
    if not (s<=n<=a+s<=w):
        raise ValueError(f'Inconsistent token counters: children={a}, steps={s}, width={w}, returned={n}')
    for key in ('generation_seconds','reset_seconds'):
        value=row.get(key,0)
        if not math.isfinite(value) or value<0: raise ValueError(f'Invalid duration: {key}')
    if n and row.get('generation_seconds',0)<=0: raise ValueError('Tokens require positive generation time')

def summarize(rows, training_seconds=0.0, pipeline_seconds=None):
    good=[r for r in rows if r.get("status")=="ok"]
    for row in good: validate_counts(row)
    def total(k):return sum(r.get(k,0) for r in good)
    a,s,w,n=map(total,("total_accept_length","total_steps","total_drafted_tokens","new_tokens"))
    gen=total("generation_seconds");reset=total("reset_seconds")
    service=gen+reset+training_seconds
    div=lambda x,y:x/y if y else None
    return dict(turns=len(good),failed_or_skipped=len(rows)-len(good),new_tokens=n,
        total_accept_length=a,total_steps=s,total_drafted_tokens=w,
        acceptance_percent=div(100*a,w),average_length=div(a,s),
        average_length_including_root=div(a+s,s),returned_tokens_per_step=div(n,s),
        discarded_verified_tokens=a+s-n,generation_seconds=gen,reset_seconds=reset,
        osd_training_seconds=training_seconds,training_inclusive_seconds=service,
        tokens_per_second_generation=div(n,gen),tokens_per_second_training_inclusive=div(n,service),
        pipeline_seconds=pipeline_seconds,tokens_per_second_pipeline=div(n,pipeline_seconds),
        trace_validation_rejected_turns=sum(r.get("adaptation",{}).get("reason")=="validation_regression" for r in good),
        trace_skipped_turns=sum(r.get("adaptation",{}).get("status")=="skipped" for r in good),
        trace_updated_turns=sum(r.get("adaptation",{}).get("status")=="updated" for r in good),
        trace_rolled_back_turns=sum(r.get("adaptation",{}).get("status")=="rolled_back" for r in good))


def gains(base, online):
    out={}
    for k in ("average_length","acceptance_percent","tokens_per_second_generation",
              "tokens_per_second_training_inclusive","tokens_per_second_pipeline"):
        a,b=base.get(k),online.get(k)
        out[k+"_relative_gain_percent"]=(100*(b/a-1) if a and b is not None else None)
    out["acceptance_delta_pp"]=(online["acceptance_percent"]-base["acceptance_percent"]
        if online.get("acceptance_percent") is not None and base.get("acceptance_percent") is not None else None)
    for suffix in ("generation","training_inclusive","pipeline"):
        k="tokens_per_second_"+suffix;a,b=base.get(k),online.get(k)
        out["speedup_"+suffix]=b/a if a and b is not None else None
    return out
