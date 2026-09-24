# -*- coding: utf-8 -*-
# Generates results2/* txt files from raw run data. Unified metrics:
#   acc%  = 100 * sum(total_accept_length) / sum(total_drafted_tokens)
#   avg   = sum(total_accept_length) / sum(total_steps)          (excludes root)
#   tps   = sum(new_tokens) / sum(generation_seconds + reset_seconds)
import json, os, unicodedata
ROOT = "/root/paddlejob/workspace/env_run/yw/adapted-ea/P0-results"
OUT  = "/root/paddlejob/workspace/env_run/yw/adapted-ea/results2"
GPU  = "NVIDIA H800"

def dw(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))
def pad(s, w, a='<'):
    s = str(s); p = max(0, w - dw(s))
    return (s + ' '*p) if a == '<' else (' '*p + s)
def table(headers, rows, aligns=None):
    n = len(headers)
    if aligns is None: aligns = ['<'] + ['>']*(n-1)
    w = [dw(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r): w[i] = max(w[i], dw(c))
    def line(r): return "  ".join(pad(r[i], w[i], aligns[i]) for i in range(n))
    out = [line(headers), "  ".join("-"*w[i] for i in range(n))]
    for r in rows: out.append(line(r))
    return "\n".join(out)

def load_rows(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if r.get("status") != "ok": continue
        rows.append(r)
    return rows
def qm(r):
    acc = 100*r["total_accept_length"]/r["total_drafted_tokens"]
    avg = r["total_accept_length"]/r["total_steps"]
    tps = r["new_tokens"]/(r["generation_seconds"]+r.get("reset_seconds",0))
    return acc, avg, tps
def pooled(rows):
    A=sum(r["total_accept_length"] for r in rows); D=sum(r["total_drafted_tokens"] for r in rows)
    S=sum(r["total_steps"] for r in rows); N=sum(r["new_tokens"] for r in rows)
    T=sum(r["generation_seconds"]+r.get("reset_seconds",0) for r in rows)
    return dict(acc=100*A/D, avg=A/S, tps=N/T, newtok=N, sec=T, n=len(rows))
def g(a, b): return (a-b)/b*100 if b else 0.0
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"

def abs_table(order, agg):
    rows = [[m, f3(agg[m]['acc']), f4(agg[m]['avg']), f2(agg[m]['tps']),
             str(agg[m]['newtok']), f1(agg[m]['sec'])] for m in order]
    return table(["方法", "接受率%", "平均接受长度", "吞吐速度(tok/s)", "吞吐总量(tok)", "时间(s)"], rows)
def f1(x): return f"{x:.1f}"
def rel_table(order, agg, base):
    b = agg[base]; rows = []
    for m in order:
        if m == base: continue
        rows.append([f"{m} vs {base}", f2(g(agg[m]['acc'], b['acc'])),
                     f2(g(agg[m]['avg'], b['avg'])), f2(g(agg[m]['tps'], b['tps']))])
    return table(["对比", "接受率提升%", "平均长度提升%", "吞吐提升%"], rows)

def max_improve(perq, order, base, keyname="qid"):
    # perq: method -> {qid: (acc,avg,tps)}
    labels = [("接受率", 0), ("平均接受长度", 1), ("吞吐速度", 2)]
    out = []
    bq = perq[base]
    for name, idx in labels:
        best = None
        for m in order:
            if m == base: continue
            for qid, vals in perq[m].items():
                if qid not in bq: continue
                bv = bq[qid][idx]; mv = vals[idx]
                if bv <= 0: continue
                gain = (mv-bv)/bv*100
                if best is None or gain > best[0]:
                    best = (gain, m, qid, bv, mv)
        if best:
            out.append(f"  {name}: {keyname}={best[2]} 方法={best[1]} "
                       f"基线={best[3]:.4f} -> {best[4]:.4f}  提升 {best[0]:+.2f}%")
    return "\n".join(out)

def write_file(path, dataset, sections):
    txt = f"dataset: {dataset}\n显卡: {GPU}\n\n" + "\n\n".join(sections) + "\n"
    open(path, "w").write(txt)
    print("wrote", path)
DS = {"AIME-2025": "aime-2025", "LongBench-Write": "longbench-write"}

def basic_dir(ds, sub): return f"{ROOT}/runs/deepseek_r1_distill_llama_8b/{ds}/seed_0/{sub}"

def gen_basic():
    for ds, slug in DS.items():
        dirs = {"baseline": basic_dir(ds,"baseline"), "Reset": basic_dir(ds,"reset"),
                "Persistent": basic_dir(ds,"persistent")}
        order = ["baseline","Reset","Persistent"]
        agg = {}; perq = {}
        for m in order:
            rows = load_rows(dirs[m]+"/stats.jsonl")
            agg[m] = pooled(rows)
            perq[m] = {r["qid"]: qm(r) for r in rows}
        params = ("实验参数:\n"
                  "  max-new-tokens: 16384\n  max-length: 32768\n  dtype: float16\n"
                  "  draft tree: total-token=60, depth=5, top-k=10\n"
                  "  scope: head_only  precision: master_fp32  warmup-steps=5, update-every=1, train-steps=1\n"
                  "  learning-rate: Reset=1e-5, Persistent=1e-6\n  seed=0, questions=0..20")
        secs = ["一、平均指标\n"+abs_table(order, agg),
                "二、相对基线提升\n"+rel_table(order, agg, "baseline"),
                "三、最大单题提升\n"+max_improve(perq, order, "baseline"),
                params]
        write_file(f"{OUT}/1-basic/{slug}-basic.txt", ds, secs)

def gen_pa():
    for ds, slug in DS.items():
        agg = {}; perq = {}
        srcs = {"baseline": basic_dir(ds,"baseline")+"/stats.jsonl",
                "Reset": basic_dir(ds,"reset")+"/stats.jsonl",
                "P": basic_dir(ds,"persistent")+"/stats.jsonl",
                "P+A": f"{ROOT}/exp3/deepseek_r1_distill_llama_8b/{ds}/seed_0/p_a/stats.jsonl"}
        order = ["baseline","Reset","P","P+A"]
        for m in order:
            rows = load_rows(srcs[m]); agg[m] = pooled(rows)
            perq[m] = {r["qid"]: qm(r) for r in rows}
        params = ("实验参数:\n"
                  "  max-new-tokens: 16384\n  max-length: 32768\n  dtype: float16\n"
                  "  draft tree: total-token=60, depth=5, top-k=10\n"
                  "  P/P+A learning-rate: 1e-6 (= 主实验 P)\n"
                  "  P+A 锚点: lambda(anchor-weight)=0.01, B(replay-mib)=0, m(replay-positions)=0, max-anchor-drift=0\n"
                  "  seed=0, questions=0..20")
        secs = ["一、平均指标\n"+abs_table(order, agg),
                "二、相对基线提升\n"+rel_table(order, agg, "baseline"),
                "三、最大单题提升\n"+max_improve(perq, order, "baseline"),
                params]
        write_file(f"{OUT}/3-p-a/{slug}-p-a.txt", ds, secs)
def gen_freeze():
    for ds, slug in DS.items():
        srcs = {"baseline": basic_dir(ds,"baseline")+"/stats.jsonl",
                "P": basic_dir(ds,"persistent")+"/stats.jsonl",
                "P-once": f"{ROOT}/exp3/deepseek_r1_distill_llama_8b/{ds}/seed_0/p_once/stats.jsonl"}
        order = ["baseline","P","P-once"]
        agg = {}; perq = {}
        for m in order:
            rows = load_rows(srcs[m]); agg[m] = pooled(rows)
            perq[m] = {r["qid"]: qm(r) for r in rows}
        params = ("实验参数:\n"
                  "  max-new-tokens: 16384\n  max-length: 32768\n  dtype: float16\n"
                  "  draft tree: total-token=60, depth=5, top-k=10\n"
                  "  learning-rate: 1e-6\n"
                  "  P-once: freeze-after-successful-updates=1 (首次成功提交后冻结), lambda=0, B=0, m=0\n"
                  "  seed=0, questions=0..20")
        secs = ["一、平均指标\n"+abs_table(order, agg),
                "二、相对基线提升\n"+rel_table(order, agg, "baseline"),
                "三、最大单题提升\n"+max_improve(perq, order, "baseline"),
                params]
        write_file(f"{OUT}/4-freeze/{slug}-freeze.txt", ds, secs)

def gen_switch():
    orders_ds = {"aime_then_lbw": "AIME(math) -> LongBench-Write(long_form)",
                 "lbw_then_aime": "LongBench-Write(long_form) -> AIME(math)"}
    for sw, desc in orders_ds.items():
        base = f"{ROOT}/exp4/deepseek_r1_distill_llama_8b/{sw}/seed_0"
        srcs = {"baseline": f"{base}/baseline/stats.jsonl", "reset": f"{base}/reset/stats.jsonl",
                "persistent": f"{base}/persistent/stats.jsonl", "p_a": f"{base}/p_a/stats.jsonl"}
        order = ["baseline","reset","persistent","p_a"]
        rowsm = {m: load_rows(srcs[m]) for m in order}
        agg = {m: pooled(rowsm[m]) for m in order}
        perq = {m: {r["stream_index"]: qm(r) for r in rowsm[m]} for m in order}
        def seg(rows, lo, hi): return pooled([r for r in rows if lo <= r["stream_index"] <= hi])
        # per-segment table
        seg_rows = []
        for m in order:
            d1 = seg(rowsm[m],0,9); d2 = seg(rowsm[m],10,19)
            seg_rows.append([m, f3(d1['acc']), f4(d1['avg']), f2(d1['tps']),
                             f3(d2['acc']), f4(d2['avg']), f2(d2['tps'])])
        seg_tbl = table(["方法","D1接受率%","D1平均长度","D1吞吐","D2接受率%","D2平均长度","D2吞吐"], seg_rows)
        params = ("实验参数:\n"
                  "  切换流: 10题domain1 + 10题domain2, 在 stream_index=10 切换\n"
                  "  max-new-tokens: 16384\n  max-length: 32768\n  dtype: float16\n"
                  "  draft tree: total-token=60, depth=5, top-k=10\n"
                  "  persistent/p_a 跨切换保留权重/优化器/anchor; reset 每题重置; baseline 不更新\n"
                  "  p_a: lambda(anchor)=0.01, B=0, m=0; learning-rate=1e-6; seed=0")
        secs = [f"顺序: {desc}",
                "一、全流平均指标\n"+abs_table(order, agg),
                "二、相对基线提升(全流)\n"+rel_table(order, agg, "baseline"),
                "三、分域指标 (D1=切换前, D2=切换后)\n"+seg_tbl,
                "四、最大单题提升\n"+max_improve(perq, order, "baseline", keyname="idx"),
                params]
        write_file(f"{OUT}/5-switch/{sw}-switch.txt", desc, secs)
def exp2_summary(path):
    s = json.load(open(path))
    return dict(acc=s["acceptance_percent"], avg=s["average_length"],
                tps=s["tokens_per_second_training_inclusive"], newtok=s["new_tokens"],
                sec=s["training_inclusive_seconds"], upd=s.get("trace_updated_turns",0),
                rb=s.get("trace_rolled_back_turns",0))
def exp2_perq(method_dir):
    import glob
    d = {}
    for f in glob.glob(method_dir+"/blocks/*/infer/turns.jsonl"):
        for l in open(f):
            l=l.strip()
            if not l: continue
            r=json.loads(l)
            if r.get("status")!="ok": continue
            acc=100*r["total_accept_length"]/r["total_drafted_tokens"]
            avg=r["total_accept_length"]/r["total_steps"]
            tps=r["new_tokens"]/(r["generation_seconds"]+r.get("reset_seconds",0))
            d[r["question_id"]]=(acc,avg,tps)
    return d

def gen_osd():
    mnt = {"AIME-2025":"8192", "LongBench-Write":"16384"}
    groups = [("osd", "EAGLE3+OSD", "EAGLE3+OSD+TraceDraft", "osd", "osd_tracedraft"),
              ("onlinespec", "ENS-EAGLE3(hedge)", "ENS-EAGLE3+TraceDraft", "onlinespec", "onlinespec_trace")]
    for ds, slug in DS.items():
        secs = []
        for gi,(grp, blab, tlab, bsub, tsub) in enumerate(groups):
            gd = f"{ROOT}/exp2/deepseek_r1_distill_llama_8b/{ds}/seed_0/{grp}"
            agg = {blab: exp2_summary(f"{gd}/{bsub}/summary.json"),
                   tlab: exp2_summary(f"{gd}/{tsub}/summary.json")}
            order = [blab, tlab]
            abst = abs_table(order, agg)
            relt = rel_table(order, agg, blab)
            perq = {blab: exp2_perq(f"{gd}/{bsub}"), tlab: exp2_perq(f"{gd}/{tsub}")}
            mx = max_improve(perq, order, blab, keyname="qid")
            upd = agg[tlab]['upd']; rb = agg[tlab]['rb']
            secs.append(f"{'一二三四五'[gi]}、{grp} 组\n{abst}\n"
                        f"  ({tlab} 更新/回退: {upd}/{rb})\n\n相对基准提升\n{relt}\n\n最大单题提升\n{mx}")
        params = ("实验参数:\n"
                  f"  max-new-tokens: {mnt[ds]}\n  max-length: 32768\n  dtype: float16\n"
                  "  draft tree: total-token=60, depth=5, top-k=10\n"
                  "  block-size=5, --no-train-last-block\n"
                  "  TraceDraft=Reset lr=1e-5 (update_round=5)\n"
                  "  OnlineSPEC: hedge temp=0.2, LR 2e-4/1e-4/4e-4, R/A=0\n  seed=0, questions=0..20")
        secs.append(params)
        write_file(f"{OUT}/2-osd-onlineSpec/{slug}-osd-onlineSpec.txt", ds, secs)

if __name__ == "__main__":
    gen_basic(); gen_pa(); gen_freeze(); gen_switch(); gen_osd()




