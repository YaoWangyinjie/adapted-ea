import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

def load(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]

def summary(rows):
    out={"n":len(rows)}
    for k in ("speed","acc_rate","avg_accept_len","time","total_accept_tokens","total_drafted_tokens","total_steps","avg_loss"):
        v=[float(r[k]) for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
        if v: out[k]={"mean":float(np.mean(v)),"median":float(np.median(v)),"p10":float(np.percentile(v,10)),"p90":float(np.percentile(v,90))}
    out["status"]={s:sum(r.get("status")==s for r in rows) for s in ("updated","skipped","rolled_back","disabled")}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("files",nargs="+")
    ap.add_argument("--output-dir",default="comparison")
    args=ap.parse_args(); data={Path(p).stem:load(p) for p in args.files}
    report={"experiments":{k:summary(v) for k,v in data.items()},"paired":{}}
    keys=set.intersection(*(set((str(r.get("qid")),r.get("choice"),r.get("turn")) for r in v) for v in data.values()))
    for name,rows in data.items():
        report["paired"][name]={"common_rows":len(keys),"groups":{}}
        groups=defaultdict(list)
        by={(str(r.get("qid")),r.get("choice"),r.get("turn")):r for r in rows}
        for key in keys:
            r=by[key]; groups[(r.get("category","unknown"),r.get("turn"))].append(r)
        report["paired"][name]["groups"]={f"{g[0]}:turn{g[1]}":summary(rs) for g,rs in groups.items()}
    baseline_name=next((k for k in data if "temperature-0.0_stats" in k and "online" not in k),None)
    if baseline_name:
        base={(str(r.get("qid")),r.get("choice"),r.get("turn")):r for r in data[baseline_name]}
        for name,rows in data.items():
            if name==baseline_name: continue
            deltas=[]
            for r in rows:
                b=base.get((str(r.get("qid")),r.get("choice"),r.get("turn")))
                if b and r.get("speed") is not None and b.get("speed") is not None:
                    deltas.append({"qid":r.get("qid"),"turn":r.get("turn"),"delta_speed":float(r["speed"])-float(b["speed"]),"delta_acc_rate":float(r.get("acc_rate",0))-float(b.get("acc_rate",0)),"delta_accept_len":float(r.get("avg_accept_len",0))-float(b.get("avg_accept_len",0))})
            report["paired"][name]["vs_baseline"]={"n":len(deltas),"mean_delta_speed":float(np.mean([x["delta_speed"] for x in deltas])) if deltas else None,"mean_delta_acc_rate":float(np.mean([x["delta_acc_rate"] for x in deltas])) if deltas else None,"mean_delta_accept_len":float(np.mean([x["delta_accept_len"] for x in deltas])) if deltas else None,"improved_speed":sum(x["delta_speed"]>0 for x in deltas)}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out/"comparison.txt").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report["experiments"],indent=2))
if __name__=="__main__": main()
